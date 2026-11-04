import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import tune_ddpm_joint as joint
import repeated_latent_validation as core
import validate_ddpm_architecture as width


class JointStudyTests(unittest.TestCase):
    def test_grid_and_plan_without_data_or_training(self):
        choices = joint.candidates()
        self.assertEqual(len(choices), 12)
        self.assertEqual(len({c['name'] for c in choices}), 12)
        self.assertEqual({c['context'] for c in choices}, {64})
        self.assertEqual(choices[0]['name'], 'reference')
        with tempfile.TemporaryDirectory() as d, patch.object(joint, 'execute') as execute:
            out = Path(d) / 'not_created'
            with contextlib.redirect_stdout(io.StringIO()) as output:
                joint.main(['--prepared', '/missing', '--structure', '/missing.csv', '--out', str(out)])
            self.assertIn('"total_inner_fits_up_to": 18', output.getvalue())
            execute.assert_not_called()
            self.assertFalse(out.exists())

    def test_shortlist_always_keeps_reference_and_deduplicates(self):
        choices = joint.candidates()
        rows = [dict(candidate=c['name'], validation_response_rmse=1. if c['name']=='reference' else .1 + i * .01)
                for i, c in enumerate(choices)]
        selected = joint.shortlist(rows, 2)
        self.assertEqual(len(selected), 3)
        self.assertIn('reference', selected)
        tied = [dict(candidate=c['name'], validation_response_rmse=.5) for c in choices]
        selected = joint.shortlist(tied, 2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0], 'reference')

    def test_candidate_optimizer_values_are_actually_applied(self):
        import torch
        pack = dict(X=torch.zeros(4, 8), Z=torch.zeros(4, 3))
        for candidate in joint.candidates():
            model, opt = width.model_for(candidate, pack, 42)
            self.assertEqual(model.head[0].out_features, candidate['hidden'])
            self.assertEqual(opt.param_groups[0]['lr'], candidate['learning_rate'])
            self.assertEqual(opt.param_groups[0]['weight_decay'], candidate['weight_decay'])

    def test_synthetic_two_stage_study_seed_coverage_and_resume(self):
        import pandas as pd
        from test_repeated_latent_validation import RepeatedLatentTests
        reduced = dict(epochs=1, validation_epochs=[1], steps=3, samples=2,
                       input_components=4, output_components=3, baseline_alphas=[10.], strengths=[0., 1.])
        with tempfile.TemporaryDirectory() as d, patch.dict(core.PROTOCOL, **reduced):
            root = Path(d)
            data_args = RepeatedLatentTests().synthetic(root)
            args = joint.argparse.Namespace(prepared=data_args.prepared, structure=data_args.structure,
                out=root/'joint', split_seed=42, folds=2, fold=0, screen_seed=42,
                repeat_seeds=[43, 44], top_k=2, threads=1, execute=True)
            with contextlib.redirect_stdout(io.StringIO()):
                joint.execute(args)
            screen = pd.read_csv(args.out/'screening.csv')
            self.assertEqual(len(screen), 12)
            finalists = json.loads((args.out/'shortlist.json').read_text())['candidates']
            self.assertIn('reference', finalists)
            trials = pd.read_csv(args.out/'trials.csv')
            for candidate, group in trials.groupby('candidate'):
                self.assertEqual(set(group.model_seed), {42, 43, 44} if candidate in finalists else {42})
            final = pd.read_csv(args.out/'finalist_summary.csv')
            self.assertTrue(final.model_seeds.eq(3).all())
            self.assertEqual(set(final.candidate), set(finalists))
            self.assertFalse(json.loads((args.out/'completed.json').read_text())['outer_test_scored'])
            self.assertFalse(json.loads((args.out/'selected_candidate.json').read_text())['outer_test_scored'])
            with patch.object(width, 'fit_trial', side_effect=AssertionError('no repeated training on resume')):
                with contextlib.redirect_stdout(io.StringIO()):
                    joint.execute(args)
            args.repeat_seeds = [45]
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                joint.execute(args)


if __name__ == '__main__':
    unittest.main()
