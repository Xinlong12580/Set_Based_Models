"""Metric-MDS geometry, rule-ID alignment, and fit-free layout plotting."""

from contextlib import redirect_stdout
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

from scripts import plot_erl_mds as mds
from tests.test_erl_serialization import fit_model


class MetricMDSTests(unittest.TestCase):
    def test_recovers_euclidean_distances_and_their_scale(self):
        for target in (np.array([[0., 7.], [7., 0.]]),
                       mds.pairwise_distances([[0, 0], [2, 0], [2, 1], [0, 1]])):
            coordinates, fit = mds.metric_mds(target, tolerance=1e-12, max_iter=3000, n_init=8)
            np.testing.assert_allclose(mds.pairwise_distances(coordinates), target, atol=1e-5)
            np.testing.assert_allclose(coordinates.mean(axis=0), 0, atol=1e-15)
            self.assertLess(fit['stress'], 1e-10)
            self.assertTrue(fit['converged'])

    def test_identical_selections_and_single_rule(self):
        for size in (1, 4):
            coordinates, fit = mds.metric_mds(np.zeros((size, size)))
            np.testing.assert_array_equal(coordinates, np.zeros((size, 2)))
            self.assertEqual(fit['stress'], 0.)
            self.assertEqual(fit['normalized_stress'], 0.)
            self.assertEqual(fit['n_iter'], 0)
            self.assertTrue(fit['converged'])
        target = mds.pairwise_distances([[0, 0], [0, 0], [1, 0], [0, 1]])
        coordinates, fit = mds.metric_mds(target, max_iter=3000, tolerance=1e-12, n_init=8)
        np.testing.assert_allclose(mds.pairwise_distances(coordinates), target, atol=1e-5)

    def test_non_euclidean_distances_have_consistent_monotonic_stress(self):
        target = np.array([[0, 1, .5, 1], [1, 0, .5, 1], [.5, .5, 0, 1], [1, 1, 1, 0]])
        coordinates, fit = mds.metric_mds(target)
        stress = np.sum((mds.pairwise_distances(coordinates) - target)**2) / 2
        self.assertGreater(stress, 0)
        self.assertAlmostEqual(stress, fit['stress'])
        self.assertAlmostEqual(np.sqrt(stress / (np.sum(target**2) / 2)), fit['normalized_stress'])
        self.assertTrue(np.all(np.diff(fit['stress_history']) <= 1e-12))
        self.assertEqual(len(fit['stress_history']), fit['n_iter'] + 1)
        self.assertEqual(fit['stress'], min(start['stress'] for start in fit['starts']))

    def test_reproducibility_multiple_starts_and_iteration_limit(self):
        target = np.ones((6, 6)) - np.eye(6)
        first, fit = mds.metric_mds(target, n_init=3, max_iter=1, seed=7)
        second, repeated = mds.metric_mds(target, n_init=3, max_iter=1, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(fit, repeated)
        self.assertEqual(fit['n_iter'], 1)
        self.assertFalse(fit['converged'])
        _, stopped = mds.metric_mds(target, tolerance=1, seed=7)
        self.assertEqual(stopped['n_iter'], 1)
        self.assertTrue(stopped['converged'])


class ERLMDSLayoutTests(unittest.TestCase):
    def test_default_retained_selection_and_explicit_rule_order(self):
        model, _ = fit_model('event')
        before = model.to_dict()
        layout = mds.fit_erl_mds(model, max_iter=10)
        ids = [rule.rule_id for rule in model.selected_rules_]
        self.assertEqual(layout['rule_ids'], ids)
        self.assertEqual(layout['selection'], 'retained')
        self.assertEqual(layout['cluster_indices'], [0] * len(ids))
        np.testing.assert_array_equal(layout['signal_jaccard'], model.signal_jaccard_[np.ix_(ids, ids)])
        ids = [len(model.candidate_rules_) - 1, 0]
        explicit = mds.fit_erl_mds(model, rule_ids=ids, cluster_indices={str(ids[0]): 3, '0': -1})
        self.assertEqual(explicit['rule_ids'], ids)
        self.assertEqual(explicit['cluster_indices'], [3, -1])
        self.assertEqual([rule['rule_id'] for rule in explicit['rules']], ids)
        self.assertEqual(model.to_dict(), before)

    def test_all_candidates_and_unpruned_default(self):
        model, _ = fit_model('event')
        layout = mds.fit_erl_mds(model, all_candidates=True, max_iter=2)
        self.assertEqual(layout['rule_ids'], list(range(len(model.candidate_rules_))))
        model.pruning_result_ = None
        model.selected_rules_ = []
        unpruned = mds.fit_erl_mds(model, max_iter=2)
        self.assertEqual(layout, unpruned)

    def test_model_and_layout_round_trip_and_clusters_do_not_change_geometry(self):
        model, _ = fit_model('event')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.json'
            model.save(path)
            ids = [0, 2, 4]
            plain = mds.fit_erl_mds(path, rule_ids=ids)
            colored = mds.fit_erl_mds(path, rule_ids=ids, cluster_indices=[0, 1, 1])
            self.assertEqual(plain['coordinates'], colored['coordinates'])
            self.assertEqual(plain['optimization'], colored['optimization'])
            self.assertEqual(plain['source_model'], str(path.resolve()))
            saved = mds.save_layout(colored, Path(temporary) / 'layout.json')
            self.assertEqual(mds.load_layout(saved), colored)
        with self.assertRaisesRegex(ValueError, 'Cluster indices'):
            mds.fit_erl_mds(model, rule_ids=[0, 1], cluster_indices=[0])

    def test_missing_matrix_and_block_diagonal_provenance(self):
        model, _ = fit_model('event')
        combined = type(model).merge([model, model])
        with self.assertWarnsRegex(UserWarning, 'assumed zero'):
            layout = mds.fit_erl_mds(combined, rule_ids=[0, len(model.candidate_rules_)])
        self.assertEqual(layout['jaccard_source'], 'block_diagonal')
        self.assertEqual(layout['signal_jaccard'], [[1., 0.], [0., 1.]])
        model.signal_jaccard_ = None
        with self.assertRaisesRegex(ValueError, 'no saved mixture Jaccard'):
            mds.fit_erl_mds(model)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib is optional')
    def test_dense_map_keeps_plot_area_and_all_points_without_label_or_legend_overflow(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        size = 123
        layout = {
            'coordinates': np.random.default_rng(42).normal(size=(size, 2)).tolist(),
            'rule_ids': list(range(size)),
            'cluster_indices': [i // 3 for i in range(120)] + [40, 41, 42],
            'clustering': {'cluster_count': 43},
            'optimization': {'normalized_stress': .3, 'converged': True, 'n_iter': 100},
            'rules': [{'rule_id': i, 'score': float(size - i)} for i in range(size)],
        }
        before = json.dumps(layout)
        for labels, expected_annotations in ((None, 8), (False, 0), (True, size)):
            with self.subTest(labels=labels):
                figure = mds.plot_mds(layout, labels=labels)
                figure.canvas.draw()
                axes = figure.axes[0]
                self.assertEqual(sum(len(group.get_offsets()) for group in axes.collections), size)
                self.assertEqual(len(axes.texts), expected_annotations)
                self.assertEqual(len(figure.legends[0].get_texts()), 10)
                self.assertGreater(axes.get_position().width, .4)
                self.assertGreater(axes.get_position().height, .6)
                renderer = figure.canvas.get_renderer()
                legend = figure.legends[0].get_window_extent(renderer)
                self.assertGreater(legend.x0, axes.get_window_extent(renderer).x1)
                self.assertGreaterEqual(legend.y0, 0)
                self.assertLessEqual(legend.y1, figure.bbox.y1)
                plt.close(figure)
        self.assertEqual(json.dumps(layout), before)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib is optional')
    def test_replot_and_recolor_never_refits_or_loads_a_model(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        model, _ = fit_model('event')
        layout = mds.fit_erl_mds(model, rule_ids=[0, 2, 4])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = mds.save_layout(layout, root / 'example.mds.json')
            clusters = root / 'clusters.json'
            clusters.write_text(json.dumps({'0': 2, '2': 5, '4': 5}))
            output = root / 'colored.png'
            updated = root / 'colored.mds.json'
            with patch.object(mds, 'metric_mds', side_effect=AssertionError('Must not fit')):
                with patch.object(type(model), 'from_dict', side_effect=AssertionError('Must not load model')):
                    with redirect_stdout(io.StringIO()):
                        result = mds.main([str(source), '--clusters', str(clusters), '--output', str(output),
                                           '--layout-output', str(updated)])
            self.assertEqual(result['coordinates'], layout['coordinates'])
            self.assertEqual(result['cluster_indices'], [2, 5, 5])
            self.assertEqual(mds.load_layout(source), layout)
            self.assertEqual(mds.load_layout(updated), result)
            self.assertTrue(output.read_bytes().startswith(b'\x89PNG'))
            figure = mds.plot_mds(result)
            self.assertEqual(len(figure.axes[0].collections), 2)
            self.assertEqual(figure.axes[0].get_aspect(), 1.)
            plt.close(figure)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib is optional')
    def test_cli_fits_from_outside_repository_without_pythonpath(self):
        model, _ = fit_model('event')
        script = Path(mds.__file__).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'example.validation.model.json'
            model.save(source)
            env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
            result = subprocess.run([sys.executable, '-B', str(script), str(source), '--rule-ids', '0', '2', '4'],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = root / 'example.validation.mds.json'
            self.assertEqual(mds.load_layout(saved)['rule_ids'], [0, 2, 4])
            self.assertTrue((root / 'example.validation.mds.png').read_bytes().startswith(b'\x89PNG'))


if __name__ == '__main__':
    unittest.main()
