"""Read-only model summaries, score rankings, weighted statistics and CLI use."""

from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import print_erl_model as printer
from set_based_models import CategoryCut, EventCut, ExistentialRule, ExistentialRuleList, RuleStep, ThresholdCut
from tests.test_erl_serialization import EVENTS, OBJECTS, fit_model


def capture(model, **options):
    output = io.StringIO()
    printer.print_model(model, stream=output, **options)
    return output.getvalue()


class ERLPrinterTests(unittest.TestCase):
    def test_default_ranks_top_retained_rules_without_changing_model(self):
        model, _ = fit_model('event')
        before = model.to_dict()
        expected = sorted(model.selected_rules_, key=lambda r: (-r.generation_score, r.rule_id))
        global_ids = [r.rule_id for r in sorted(model.candidate_rules_, key=lambda r: (-r.generation_score, r.rule_id))]
        text = capture(model, top=2)
        for rank, rule in enumerate(expected[:2], 1):
            self.assertIn(f'[{rank}] rule={rule.rule_id} global_rank={global_ids.index(rule.rule_id) + 1}', text)
        self.assertEqual(sum(line.startswith('[') for line in text.splitlines()), min(2, len(expected)))
        self.assertIn('retained rules', text)
        self.assertNotIn('Prefix history', text)
        self.assertEqual(model.to_dict(), before)

    def test_verbosity_summary_details_and_full_configuration(self):
        model, _ = fit_model('combined')
        summary = capture(model, verbosity=0)
        self.assertIn('Candidates:', summary)
        self.assertNotIn('Top ', summary)
        self.assertNotIn('EXISTS', summary)
        detailed = capture(model, verbosity=2, top=1)
        self.assertIn('Prefix history', detailed)
        self.assertIn('Accepted/total weights:', detailed)
        self.assertIn('Event Jaccard:', detailed)
        self.assertNotIn('Container configuration:', detailed)
        verbose = capture(model, verbosity=3, top=1)
        self.assertIn('Container configuration:', verbose)
        self.assertIn('Object schema:', verbose)
        self.assertIn('Event schema:', verbose)
        self.assertIn('guard=isolation_valid', verbose)

    def test_all_candidates_requested_ids_and_unpruned_or_empty_selection(self):
        model, _ = fit_model('object')
        text = capture(model, all_candidates=True, top=0)
        self.assertEqual(sum(line.startswith('[') for line in text.splitlines()), len(model.candidate_rules_))
        discarded = next(r for r in model.candidate_rules_ if r.rule_id not in {s.rule_id for s in model.selected_rules_})
        requested = capture(model, rule_ids=[discarded.rule_id])
        self.assertIn(f'[1] rule={discarded.rule_id}', requested)
        self.assertNotIn('status=retained', requested)
        model.pruning_result_ = None
        model.selected_rules_ = []
        self.assertIn('all candidates', capture(model, top=1))
        self.assertIn('status=unpruned', capture(model, top=1))
        model.object_cut_priority = ('pt',) * 10
        model.prune_candidates(require_complete_priority=True, similarity_filter=False)
        text = capture(model)
        self.assertIn('Final selection: 0 retained', text)
        self.assertIn('No rules in this pool', text)
        self.assertFalse(any(line.startswith('[') for line in text.splitlines()))
        empty = ExistentialRuleList((), event_observables=EVENTS)
        self.assertIn('Candidates: 0', capture(empty))

    def test_normalized_guards_categories_same_slot_counts_and_weighted_acceptance(self):
        fitted, _ = fit_model('combined')
        stats = fitted.candidate_rules_[0].statistics
        stats = replace(stats, signal_accepted=1, signal_total=4, signal_weight_accepted=2., signal_weight_total=10.,
                        background_accepted=1, background_total=5, background_weight_accepted=1., background_weight_total=15.,
                        minimum_count=2, rule_depth=4)
        cuts = (CategoryCut(1, {1}), ThresholdCut(2, '<=', .3),
                EventCut(ThresholdCut(3, '>', 1.5)), EventCut(ThresholdCut(3, '>', .5)))
        rule = ExistentialRule(0, OBJECTS, tuple(RuleStep(i, cut, stats, .3) for i, cut in enumerate(cuts, 1)),
                               minimum_count=2, event_observables=EVENTS)
        model = ExistentialRuleList(OBJECTS, event_observables=EVENTS)
        model.candidate_rules_ = [rule]
        text = capture(model, verbosity=1)
        self.assertIn('depth=4 simplified_cuts=3', text)
        self.assertIn('event: count >= 2', text)
        self.assertNotIn('count >= 1', text)
        self.assertIn('COUNT >= 2 matching slots', text)
        self.assertIn('type_id in {photon}', text)
        self.assertIn('isolation_valid == True AND isolation <= 0.3', text)
        self.assertIn('mixture=1/4 (20%)', text)
        self.assertIn('reference=1/5 (6.66667%)', text)

    def test_json_inputs_skip_dense_cache_reconstruction_and_never_run_inference(self):
        model, _ = fit_model('event')
        payload = model.to_dict()
        before = json.dumps(payload)
        with patch.object(ExistentialRuleList, 'from_dict', wraps=ExistentialRuleList.from_dict) as load:
            with patch.object(ExistentialRuleList, 'fit', side_effect=AssertionError('Must not train')):
                with patch.object(ExistentialRule, 'predict', side_effect=AssertionError('Must not infer')):
                    with patch.object(ExistentialRuleList, 'prune_candidates', side_effect=AssertionError('Must not prune')):
                        text = capture(payload, top=1, verbosity=2)
        self.assertNotIn('event_jaccard', load.call_args.args[0])
        self.assertIn('Event Jaccard: mixture and reference', text)
        self.assertEqual(json.dumps(payload), before)

    def test_merged_and_self_pruned_provenance_is_reported(self):
        first, _ = fit_model('event')
        merged = ExistentialRuleList.merge([first, first], source_names=['one.json', 'two.json'])
        text = capture(merged, verbosity=3, rule_ids=[len(first.candidate_rules_)])
        self.assertIn('compare them after common-data recalculation', text)
        self.assertIn('source=1 model=two.json original_rule=0', text)
        self.assertIn('Source models and their search configurations:', text)
        first.self_pruning_provenance_ = {
            'configuration': {'stricter_score_improvement': .001},
            'source_rule_ids': list(range(12)), 'evaluated_variant_count': 50,
            'locally_retained_variant_count': 20,
            'rules': [{'parent_rule_ids': list(range(12))} for _ in first.candidate_rules_],
        }
        detailed = capture(first, verbosity=2, top=1)
        self.assertIn('12 source rules; 50 evaluated variants; 20 local survivors', detailed)
        self.assertIn('(+4 more)', detailed)
        self.assertIn(str(list(range(12))), capture(first, verbosity=3, top=1))

    def test_cli_works_outside_repository_and_preserves_model_bytes(self):
        model, _ = fit_model('event')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = model.save(root / 'example.model.json')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
            result = subprocess.run([sys.executable, '-B', str(Path(printer.__file__).resolve()), str(source),
                                     '--top', '1', '--verbosity', '2'], cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Top 1 of', result.stdout)
            self.assertIn('Prefix history', result.stdout)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            with redirect_stdout(io.StringIO()) as output:
                printer.main([str(source), '-v', '0'])
            self.assertNotIn('Top ', output.getvalue())


if __name__ == '__main__':
    unittest.main()
