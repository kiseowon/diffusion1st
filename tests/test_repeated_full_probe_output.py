import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import repeat_time_structure_pilot as repeat
import repeated_latent_validation as core

class RepeatedFullProbeTests(unittest.TestCase):
    def test_nine_condition_plan_no_io(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            count=repeat.main(['--prepared','/absent','--structure','/absent','--smiles','/absent','--out','/absent',
                '--input-space','pca','--output-space','full','--arms','basic','morgan','descriptor',
                '--cache-diffusion-context'])
        self.assertEqual(count,135)
        plan=json.loads(output.getvalue())
        self.assertEqual(plan['fits_up_to'],270)
        self.assertEqual(plan['output_space'],'full')
        self.assertTrue(plan['cache_diffusion_context'])

    def test_full_output_pooled_three_arm_study_and_resume(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root)
            smiles=root/'smiles.csv'
            pd.DataFrame(dict(compound=[f'compound_{i:02}' for i in range(20)],
                smiles=['C'*(i+2) for i in range(20)])).to_csv(smiles,index=False)
            argv=['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),
                '--out',str(root/'repeat'),'--input-space','pca','--output-space','full',
                '--arms','basic','morgan','descriptor','--cache-diffusion-context',
                '--epochs','1','--folds','2','--model-seeds','42','43','--threads','1','--execute']
            with contextlib.redirect_stdout(io.StringIO()):repeat.main(argv)
            reg=pd.read_csv(root/'repeat/pooled_by_model_seed.csv')
            self.assertEqual(len(reg),20)
            self.assertEqual(len(reg[reg.model_seed.eq(-1)]),2)
            self.assertTrue(reg.n_targets.eq(40).all())
            self.assertEqual(len(pd.read_csv(root/'repeat/summary.csv')),11)
            pair=pd.read_csv(root/'repeat/paired_rmse_differences.csv')
            for label in ['descriptor_minus_basic_ddpm','descriptor_minus_morgan_mlp',
                          'morgan_ddpm_minus_mlp','basic_cgan_minus_time_level_mean']:
                self.assertEqual(len(pair[pair.comparison.eq(label)]),2)
            unit=json.loads((root/'repeat/fold_0/seed_42/run.json').read_text())
            self.assertTrue(unit['settings']['cache_diffusion_context'])
            self.assertFalse(unit['settings']['output_pca'])
            status=json.loads((root/'repeat/evaluation_status.json').read_text())
            self.assertEqual(status['output_space'],'full')
            self.assertTrue(status['all_requested_units_completed'])
            with patch.object(core,'fit_job',side_effect=AssertionError('No new training on resume')):
                with contextlib.redirect_stdout(io.StringIO()):repeat.main(argv)
