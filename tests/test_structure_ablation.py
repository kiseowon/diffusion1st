import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import repeated_latent_validation as core
import compare_structure_ablation as compare


class StructureAblationTests(unittest.TestCase):
    def test_off_plan_has_no_io(self):
        with tempfile.TemporaryDirectory() as d, patch.object(core, 'execute') as execute:
            out = Path(d) / 'absent'
            with contextlib.redirect_stdout(io.StringIO()) as output:
                core.main(['--prepared', '/missing', '--structure', '/missing.csv', '--out', str(out), '--no-structure'])
            self.assertIn('"structure": false', output.getvalue())
            execute.assert_not_called()
            self.assertFalse(out.exists())

    def test_only_morgan_and_presence_slots_change_and_model_shapes_match(self):
        import numpy as np
        import torch
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            args = RepeatedLatentTests().synthetic(Path(d))
            first, data, ids, meta, Y, _, _ = core.load_data(args)
            fit = np.arange(16)
            baseline, _ = core.baseline_stage(meta, Y, fit, 10)
            on = core.prepare_stage(first, data, ids, meta, Y, fit, baseline)
            off = core.prepare_stage(first, data, ids, meta, Y, fit, baseline, use_structure=False)
            n = data.structure.shape[1]
            torch.testing.assert_close(on['X'][:, :-n], off['X'][:, :-n], rtol=0, atol=0)
            self.assertTrue((off['X'][:, -n:] == 0).all())
            self.assertTrue((on['X'][:, -1] == 1).all())
            torch.testing.assert_close(on['Z'], off['Z'], rtol=0, atol=0)
            self.assertEqual(on['state']['output_scale'], off['state']['output_scale'])
            self.assertFalse(off['state']['structure_enabled'])
            for kind in ['mlp', 'cgan', 'ddpm']:
                a, _, da, _ = core.make_family_model(kind, on['X'].shape[1], on['Z'].shape[1], 42)
                b, _, db, _ = core.make_family_model(kind, off['X'].shape[1], off['Z'].shape[1], 42)
                for key in a.state_dict():
                    self.assertTrue(torch.equal(a.state_dict()[key], b.state_dict()[key]))
                self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
                if da is not None:
                    for key in da.state_dict():
                        self.assertTrue(torch.equal(da.state_dict()[key], db.state_dict()[key]))

    def test_synthetic_completed_on_off_comparison_and_resume_mismatch(self):
        import pandas as pd
        from types import SimpleNamespace
        from test_repeated_latent_validation import RepeatedLatentTests
        reduced = dict(epochs=1, validation_epochs=[1], steps=3, samples=2,
                       input_components=4, output_components=3, baseline_alphas=[10.], strengths=[0., 1.])
        with tempfile.TemporaryDirectory() as d, patch.dict(core.PROTOCOL, **reduced):
            root = Path(d)
            args = RepeatedLatentTests().synthetic(root)
            args.families = ['mlp', 'cgan', 'ddpm']
            on_run = args.out
            with contextlib.redirect_stdout(io.StringIO()):
                core.execute(args)
            args.out = root / 'off'; args.no_structure = True; args.paired_on_run = on_run
            with contextlib.redirect_stdout(io.StringIO()):
                core.execute(args)
            off_run = args.out
            cargs = SimpleNamespace(on_run=on_run, off_run=off_run, out=root/'comparison', bootstrap_samples=30)
            with contextlib.redirect_stdout(io.StringIO()):
                compare.execute(cargs)
            summary = pd.read_csv(cargs.out/'summary.csv')
            self.assertEqual(set(summary.model), {'mlp', 'cgan', 'ddpm'})
            paired = pd.read_csv(cargs.out/'paired_by_split_model_seed.csv')
            self.assertEqual(len(paired), 3)
            self.assertTrue((cargs.out/'compound_bootstrap.csv').is_file())
            on = json.loads((on_run/'run.json').read_text())
            off = json.loads((off_run/'run.json').read_text())
            for mutation in ['versions', 'input_sha256', 'protocol']:
                changed = copy.deepcopy(off)
                if mutation == 'versions': changed['versions']['torch'] = 'different'
                elif mutation == 'input_sha256': changed[mutation][next(iter(changed[mutation]))] = 'changed'
                else: changed['protocol']['epochs'] = 999
                with self.assertRaises(ValueError): compare.paired_contract(on, changed)
            legacy = copy.deepcopy(on)
            key = next(k for k in legacy['source_sha256'] if Path(k).name == 'repeated_latent_validation.py')
            legacy['source_sha256'][key] = compare.LEGACY_ON_RUNNER_SHA256
            legacy['arguments'].pop('no_structure')
            compare.paired_contract(legacy, off)
            args.no_structure = False
            with self.assertRaisesRegex(ValueError, 'only for a structure-off'): core.execute(args)
            args.paired_on_run = None
            with self.assertRaisesRegex(ValueError, 'contract differs'): core.execute(args)


if __name__ == '__main__':
    unittest.main()
