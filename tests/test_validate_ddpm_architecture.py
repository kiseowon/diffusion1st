import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import validate_ddpm_architecture as study
import repeated_latent_validation as core


class ArchitectureTests(unittest.TestCase):
    def test_plan_does_not_read_or_train(self):
        with tempfile.TemporaryDirectory() as d, patch.object(study, 'execute') as execute:
            out = Path(d) / 'absent'
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                study.main(['--prepared', '/missing', '--structure', '/missing.csv', '--out', str(out)])
            self.assertIn('"inner_fits": 15', printed.getvalue())
            execute.assert_not_called()
            self.assertFalse(out.exists())

    def test_reference_network_matches_original_initialization_exactly(self):
        import torch
        pack = dict(X=torch.zeros(5, 20), Z=torch.zeros(5, 8))
        original, _ = core.make_model(20, 8, 42)
        model, _ = study.model_for(study.CANDIDATES[0], pack, 42)
        self.assertEqual(set(original.state_dict()), set(model.state_dict()))
        for key, expected in original.state_dict().items():
            self.assertTrue(torch.equal(expected, model.state_dict()[key]), key)
        x, noisy, t = torch.randn(3, 20), torch.randn(3, 8), torch.tensor([0, 1, 2])
        torch.testing.assert_close(original(x, noisy, t), model(x, noisy, t), rtol=0, atol=0)

    def test_sensitivity_candidates_only_change_one_width(self):
        import torch
        pack = dict(X=torch.zeros(5, 20), Z=torch.zeros(5, 8))
        for candidate in study.CANDIDATES:
            model, _ = study.model_for(candidate, pack, 42)
            self.assertEqual(model.context[0].out_features, candidate['context'])
            self.assertEqual(model.head[0].out_features, candidate['hidden'])
            self.assertEqual(model.head[-1].out_features, 8)
            if candidate['name'] != 'reference':
                self.assertEqual(int(candidate['context'] != 64) + int(candidate['hidden'] != 128), 1)

    def test_outer_test_truth_does_not_change_validation_selection(self):
        import numpy as np
        import pandas as pd
        from test_repeated_latent_validation import RepeatedLatentTests
        reduced = dict(epochs=1, validation_epochs=[1], steps=3, samples=2,
                       input_components=4, output_components=3, baseline_alphas=[10.], strengths=[0., 1.])
        with tempfile.TemporaryDirectory() as d, patch.dict(core.PROTOCOL, **reduced):
            root = Path(d)
            original_args = RepeatedLatentTests().synthetic(root)
            args = study.argparse.Namespace(prepared=original_args.prepared, structure=original_args.structure,
                out=root / 'study1', split_seed=42, folds=2, fold=0, model_seeds=[42], threads=1, execute=True)
            with contextlib.redirect_stdout(io.StringIO()):
                study.execute(args)
            summary1 = pd.read_csv(args.out / 'candidate_summary.csv')
            self.assertEqual(len(summary1), 5)
            completed = json.loads((args.out / 'completed.json').read_text())
            self.assertEqual(completed['completed_trials'], 5)
            self.assertFalse(completed['outer_test_scored'])
            table = pd.read_csv(args.prepared / 'vivo_conditions.csv')
            mapping = core.compound_folds(table.compound.unique(), 42, 2)
            heldout = table.compound.map(mapping).eq(0) & table.level.ne('Control')
            file = args.prepared / 'vivo_expression.npy'
            target = np.load(file); target[heldout] += 1000; np.save(file, target)
            args.out = root / 'study2'
            with contextlib.redirect_stdout(io.StringIO()):
                study.execute(args)
            summary2 = pd.read_csv(args.out / 'candidate_summary.csv')
            pd.testing.assert_frame_equal(summary1, summary2, check_exact=True)
            with patch.object(study, 'fit_trial', side_effect=AssertionError('resume must skip fits')):
                with contextlib.redirect_stdout(io.StringIO()):
                    study.execute(args)


if __name__ == '__main__':
    unittest.main()
