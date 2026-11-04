import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_repeated_predictions as evaluate


class EvaluationTests(unittest.TestCase):
    def test_perfect_predictions_and_undefined_constant_correlations(self):
        import numpy as np
        y = np.array([[-2., 0., 2.], [-1., .2, 1.], [0., 0., 0.]])
        reg, cls, _, detail = evaluate.score(y, y.copy(), [1.])
        self.assertEqual(reg['rmse'], 0)
        self.assertEqual(reg['mae'], 0)
        self.assertEqual(reg['r2_probe_variance_weighted'], 1)
        self.assertEqual(reg['pearson_valid_targets'], 2)
        self.assertAlmostEqual(reg['pearson_target_mean'], 1)
        self.assertEqual(cls[0]['macro_f1'], 1)
        self.assertEqual(cls[0]['accuracy'], 1)
        self.assertTrue(np.isnan(detail['pearson'][-1]))

    def test_classification_matches_fixed_three_class_sklearn_definition(self):
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score
        y = np.array([[-2., 0., 2., .1], [-1., 0., 1., .1]])
        p = np.array([[0., 0., .1, 0.], [-2., 1., 2., 0.]])
        _, cls, confusion, _ = evaluate.score(y, p, [1.])
        truth = np.where(y >= 1, 2, np.where(y <= -1, 0, 1)).ravel()
        pred = np.where(p >= 1, 2, np.where(p <= -1, 0, 1)).ravel()
        self.assertAlmostEqual(cls[0]['macro_f1'], f1_score(truth, pred, labels=[0, 1, 2], average='macro', zero_division=0))
        self.assertAlmostEqual(cls[0]['accuracy'], accuracy_score(truth, pred))
        self.assertEqual(sum(x['count'] for x in confusion), y.size)
        # +0.1 has the right sign for +2, but predicts the 'small' amplitude class.
        self.assertEqual(cls[0]['active_sign_accuracy'], .75)

    def test_saved_predictions_evaluated_without_training_and_input_change_rejected(self):
        import pandas as pd
        from types import SimpleNamespace
        from test_repeated_latent_validation import RepeatedLatentTests
        import repeated_latent_validation as run
        reduced = dict(epochs=1, validation_epochs=[1], steps=3, samples=2,
                       input_components=4, output_components=3, baseline_alphas=[10.], strengths=[0., 1.])
        with tempfile.TemporaryDirectory() as d, patch.dict(run.PROTOCOL, **reduced):
            root = Path(d)
            args = RepeatedLatentTests().synthetic(root)
            with contextlib.redirect_stdout(io.StringIO()):
                run.execute(args)
            evaluation = SimpleNamespace(run=args.out, out=root / 'evaluation', thresholds=[.5, 1.])
            with patch.object(run, 'train_epoch', side_effect=AssertionError('evaluation must not train')):
                with contextlib.redirect_stdout(io.StringIO()):
                    evaluate.execute(evaluation)
            record = json.loads((evaluation.out / 'evaluation.json').read_text())
            self.assertFalse(record['training_performed'])
            self.assertFalse(record['checkpoint_or_joblib_loaded'])
            table = pd.read_csv(evaluation.out / 'regression_by_unit.csv')
            self.assertEqual(len(table), 5)
            self.assertTrue(table.n_targets.eq(40).all())
            cls = pd.read_csv(evaluation.out / 'classification_by_unit.csv')
            self.assertEqual(len(cls), 10)
            with self.assertRaisesRegex(ValueError, 'new evaluation'):
                evaluate.execute(evaluation)
            source = args.prepared / 'vivo_conditions.csv'
            source.write_bytes(source.read_bytes() + b'\n')
            evaluation.out = root / 'bad_eval'
            with self.assertRaisesRegex(ValueError, 'checksum changed'):
                evaluate.execute(evaluation)
            self.assertFalse(evaluation.out.exists())

    def test_three_family_saved_prediction_evaluation(self):
        import pandas as pd
        from types import SimpleNamespace
        from test_repeated_latent_validation import RepeatedLatentTests
        import repeated_latent_validation as run
        reduced = dict(epochs=1, validation_epochs=[1], steps=3, samples=2,
                       input_components=4, output_components=3, baseline_alphas=[10.], strengths=[0., 1.])
        with tempfile.TemporaryDirectory() as d, patch.dict(run.PROTOCOL, **reduced):
            root = Path(d)
            args = RepeatedLatentTests().synthetic(root)
            args.families = ['mlp', 'cgan', 'ddpm']
            with contextlib.redirect_stdout(io.StringIO()):
                run.execute(args)
            self.assertEqual(json.loads((args.out / 'completed.json').read_text())['completed_jobs'], 6)
            paired = pd.read_csv(args.out / 'paired_comparisons.csv')
            self.assertIn('ddpm_minus_cgan', paired.columns)
            self.assertIn('cgan_minus_mlp', paired.columns)
            evaluation = SimpleNamespace(run=args.out, out=root / 'evaluation', thresholds=[.5, 1.])
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate.execute(evaluation)
            table = pd.read_csv(evaluation.out / 'regression_by_unit.csv')
            self.assertEqual(len(table), 6)
            self.assertIn('cgan', set(table.model))


if __name__ == '__main__':
    unittest.main()
