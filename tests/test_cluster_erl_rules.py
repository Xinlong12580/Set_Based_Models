"""Clustering on original overlaps, singleton retention and visualizer integration."""

from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import cluster_erl_rules as clustering
from scripts import plot_erl_mds as visualizer
from tests.test_erl_serialization import fit_model


def example_layout():
    similarity = np.eye(6)
    similarity[0, 1] = similarity[1, 0] = .9
    similarity[0, 2] = similarity[2, 0] = .8
    similarity[1, 2] = similarity[2, 1] = .85
    similarity[3, 4] = similarity[4, 3] = .95
    ids = [10, 20, 30, 40, 50, 60]
    return {
        'format': visualizer.FORMAT, 'schema_version': 1, 'rule_ids': ids,
        'signal_jaccard': similarity.tolist(), 'coordinates': np.zeros((6, 2)).tolist(),
        'cluster_indices': [0] * 6,
        'optimization': {'normalized_stress': .2, 'n_iter': 10, 'converged': True},
        'rules': [{'rule_id': rule_id, 'score': score, 'rule': str(rule_id),
                   'signal_acceptance': .1, 'reference_acceptance': .001}
                  for rule_id, score in zip(ids, [.2, .3, .3, .1, .4, .05])],
    }


class CompleteLinkageTests(unittest.TestCase):
    def test_clusters_and_inclusive_threshold_without_chaining(self):
        # Jaccard distances for {1,2}, {1,2,3}, {2,3}: two close edges form a chain.
        distances = np.array([[0, 1/3, 2/3], [1/3, 0, 1/3], [2/3, 1/3, 0]])
        before = distances.copy()
        labels, merges = clustering.complete_linkage(distances, .5)
        np.testing.assert_array_equal(labels, [0, 0, 1])
        self.assertEqual(merges, [{'node': 3, 'left': 0, 'right': 1, 'distance': 1/3, 'size': 2}])
        np.testing.assert_array_equal(distances, before)
        labels, _ = clustering.complete_linkage([[0, .5], [.5, 0]], .5)
        np.testing.assert_array_equal(labels, [0, 0])

    def test_identical_disjoint_single_and_empty_inputs(self):
        labels, _ = clustering.complete_linkage(np.zeros((4, 4)), 0)
        np.testing.assert_array_equal(labels, [0, 0, 0, 0])
        distances = np.ones((4, 4)) - np.eye(4)
        labels, merges = clustering.complete_linkage(distances, .99)
        np.testing.assert_array_equal(labels, [0, 1, 2, 3])
        self.assertEqual(merges, [])
        labels, _ = clustering.complete_linkage(distances, 1)
        np.testing.assert_array_equal(labels, [0, 0, 0, 0])
        np.testing.assert_array_equal(clustering.complete_linkage([[0]])[0], [0])
        self.assertEqual(clustering.complete_linkage(np.empty((0, 0)))[0].size, 0)

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy is optional, used only as a test oracle')
    def test_matches_scipy_on_independent_distances(self):
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        rng = np.random.default_rng(42)
        for size in (2, 3, 12, 40):
            distances = rng.random((size, size))
            distances = (distances + distances.T) / 2
            np.fill_diagonal(distances, 0)
            hierarchy = linkage(squareform(distances), method='complete')
            for threshold in (0, .25, .5, .9, 1):
                labels, _ = clustering.complete_linkage(distances, threshold)
                expected = fcluster(hierarchy, threshold, criterion='distance')
                np.testing.assert_array_equal(labels[:, None] == labels[None, :],
                                              expected[:, None] == expected[None, :])


class ERLClusteringTests(unittest.TestCase):
    def test_metadata_alignment_representatives_and_unchanged_coordinates(self):
        source = example_layout()
        before = copy.deepcopy(source)
        result = clustering.cluster_layout(source, distance_threshold=.3)
        self.assertEqual(source, before)
        self.assertEqual(result['coordinates'], source['coordinates'])
        self.assertEqual(result['optimization'], source['optimization'])
        self.assertEqual(result['cluster_indices'], [0, 0, 0, 1, 1, 2])
        clusters = result['clustering']['clusters']
        self.assertEqual([group['rule_ids'] for group in clusters], [[10, 20, 30], [40, 50], [60]])
        self.assertEqual([group['representative_rule_id'] for group in clusters], [20, 50, 60])
        self.assertAlmostEqual(clusters[0]['minimum_jaccard'], .8)
        self.assertAlmostEqual(clusters[0]['mean_jaccard'], .85)
        self.assertIsNone(clusters[-1]['minimum_jaccard'])
        self.assertEqual(result['clustering']['singleton_count'], 1)

    def test_changing_coordinates_never_changes_clusters_and_threshold_recuts(self):
        source = example_layout()
        original = clustering.cluster_layout(source, distance_threshold=.3)
        source['coordinates'] = np.random.default_rng(1).normal(size=(6, 2)).tolist()
        changed = clustering.cluster_layout(source, distance_threshold=.3)
        self.assertEqual(original['clustering'], changed['clustering'])
        tighter = clustering.cluster_layout(original, distance_threshold=0)
        self.assertEqual(tighter['cluster_indices'], list(range(6)))
        self.assertEqual(original['cluster_indices'], [0, 0, 0, 1, 1, 2])

    def test_core_model_input_and_mapping_round_trip(self):
        model, _ = fit_model('event')
        before = model.to_dict()
        result = clustering.cluster_erl_model(model, max_iter=3)
        self.assertEqual(result['rule_ids'], [rule.rule_id for rule in model.selected_rules_])
        self.assertEqual(model.to_dict(), before)
        for cluster in result['clustering']['clusters']:
            if cluster['size'] > 1:
                self.assertGreaterEqual(cluster['minimum_jaccard'], .5)
        with tempfile.TemporaryDirectory() as temporary:
            path = clustering.save_clusters(result, Path(temporary) / 'clusters.json')
            mapping = json.loads(path.read_text())
            self.assertEqual(visualizer._cluster_indices(result['rule_ids'], mapping), result['cluster_indices'])

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib is optional')
    def test_saved_layout_cli_calls_visualizer_without_optimization(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        layout = example_layout()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = visualizer.save_layout(layout, root / 'example.mds.json')
            prefix = root / 'result'
            with patch.object(visualizer, 'metric_mds', side_effect=AssertionError('Must not fit')):
                with patch.object(visualizer, 'plot_mds', wraps=visualizer.plot_mds) as plot:
                    with redirect_stdout(io.StringIO()):
                        result = clustering.main([str(path), '--distance-threshold', '.3', '--output-prefix', str(prefix),
                                                  '--no-labels'])
            plot.assert_called_once()
            self.assertIs(plot.call_args.kwargs['labels'], False)
            self.assertEqual(visualizer.load_layout(path), layout)
            self.assertEqual(visualizer.load_layout(root / 'result.clustered.mds.json'), result)
            self.assertTrue((root / 'result.clustered.mds.png').read_bytes().startswith(b'\x89PNG'))
            figure = visualizer.plot_mds(result)
            # Two non-singleton groups and one combined gray singleton collection.
            self.assertEqual(len(figure.axes[0].collections), 3)
            plt.close(figure)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib is optional')
    def test_model_cli_from_outside_repository(self):
        model, _ = fit_model('event')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'example.validation.model.json'
            model.save(path)
            env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
            result = subprocess.run([sys.executable, '-B', str(Path(clustering.__file__).resolve()), str(path),
                                     '--rule-ids', '0', '2', '4', '--max-iter', '2'],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = visualizer.load_layout(root / 'example.validation.clustered.mds.json')
            self.assertEqual(output['rule_ids'], [0, 2, 4])
            self.assertEqual(len(output['cluster_indices']), 3)
            self.assertTrue((root / 'example.validation.clustered.mds.png').exists())


if __name__ == '__main__':
    unittest.main()
