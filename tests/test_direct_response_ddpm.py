import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import repeated_latent_validation as core
import evaluate_repeated_predictions as evaluate


class DirectResponseTests(unittest.TestCase):
    def test_plan_has_no_ridge_or_strength_selection_and_no_training(self):
        with tempfile.TemporaryDirectory() as d, patch.object(core, 'execute') as execute:
            out = Path(d)/'absent'
            with contextlib.redirect_stdout(io.StringIO()) as output:
                jobs = core.main(['--prepared', '/missing', '--structure', '/missing.csv', '--out', str(out),
                                  '--families', 'ddpm', '--prediction-target', 'direct_response'])
            plan = json.loads(output.getvalue())
            self.assertEqual(len(jobs), 15)
            self.assertEqual(plan['protocol']['strengths'], [1.])
            self.assertEqual(plan['protocol']['residual'], 'none_no_ridge')
            self.assertNotIn('baseline_alphas', plan['protocol'])
            self.assertFalse(out.exists())
            execute.assert_not_called()

    def test_output_pca_is_response_not_residual_and_nonzero_offset_rejected(self):
        import numpy as np
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            args = RepeatedLatentTests().synthetic(Path(d))
            first, data, ids, meta, Y, _, _ = core.load_data(args)
            fit = np.arange(16)
            pack = core.prepare_stage(first, data, ids, meta, Y, fit, np.zeros_like(Y), prediction_target='direct_response')
            np.testing.assert_allclose(pack['state']['output_pca'].mean_, Y[fit].mean(0), rtol=0, atol=1e-12)
            self.assertEqual(pack['state']['prediction_target'], 'direct_response')
            with self.assertRaisesRegex(ValueError, 'nonzero'):
                core.prepare_stage(first, data, ids, meta, Y, fit, np.ones_like(Y), prediction_target='direct_response')

    def test_direct_saved_predictions_evaluate_without_any_ridge_fit(self):
        import pandas as pd
        from types import SimpleNamespace
        from test_repeated_latent_validation import RepeatedLatentTests
        reduced = dict(epochs=1, validation_epochs=[1], steps=3, samples=2,
                       input_components=4, output_components=3, baseline_alphas=[10.], strengths=[0., .5, 1.])
        with tempfile.TemporaryDirectory() as d, patch.dict(core.PROTOCOL, **reduced):
            root = Path(d)
            args = RepeatedLatentTests().synthetic(root)
            args.prediction_target='direct_response'; args.families=['ddpm']
            with patch.object(core, 'baseline_stage', side_effect=AssertionError('Ridge must never fit')) as ridge:
                with contextlib.redirect_stdout(io.StringIO()):
                    core.execute(args)
                ridge.assert_not_called()
            self.assertFalse(list(args.out.rglob('ridge.joblib')))
            result = pd.read_csv(args.out/'results.csv')
            self.assertEqual(len(result), 2)
            self.assertTrue(result.strength.eq(1.).all())
            self.assertTrue(result.epoch.ge(1).all())
            self.assertTrue(result.prediction_target.eq('direct_response').all())
            summary = pd.read_csv(args.out/'summary.csv')
            self.assertEqual(set(summary.model), {'ddpm', 'time_level_mean', 'zero_response'})
            eargs=SimpleNamespace(run=args.out,out=root/'evaluation',thresholds=[.5,1.])
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate.execute(eargs)
            evaluation = pd.read_csv(eargs.out/'regression_by_unit.csv')
            self.assertEqual(len(evaluation), 3)
            self.assertNotIn('target_conditions_ridge', set(evaluation.model))
            self.assertTrue(evaluation.prediction_target.eq('direct_response').all())
            self.assertTrue(evaluation.rmse_improvement_vs_ridge_pct.isna().all())


if __name__ == '__main__':
    unittest.main()
