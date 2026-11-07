import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import repeat_time_structure_pilot as repeat
import repeated_latent_validation as core

class RepeatTimeTests(unittest.TestCase):
    def test_default_plan_no_io(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(repeat.main(['--prepared','/absent','--structure','/absent','--smiles','/absent','--out','/absent']),90)

    def test_full_fold_pooled_metrics_baseline_not_repeated_and_resume(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root)
            smiles=root/'smiles.csv'
            pd.DataFrame(dict(compound=[f'compound_{i:02}' for i in range(20)],smiles=['CC']*20)).to_csv(smiles,index=False)
            argv=['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),
                '--out',str(root/'repeat'),'--epochs','1','--folds','2','--model-seeds','42','43',
                '--families','mlp','ddpm','--threads','1','--execute']
            with contextlib.redirect_stdout(io.StringIO()):repeat.main(argv)
            table=pd.read_csv(root/'repeat/pooled_by_model_seed.csv')
            self.assertEqual(len(table),10)
            self.assertTrue(table.n_targets.eq(40).all())
            self.assertEqual(len(table[table.model_seed.eq(-1)]),2)
            self.assertEqual(len(list((root/'repeat').rglob('complete.json'))),16)
            detail=pd.read_csv(root/'repeat/oof_condition_metrics.csv')
            self.assertFalse(detail.duplicated(['condition_id','arm','model','model_seed']).any())
            with patch.object(core,'fit_job',side_effect=AssertionError('No new training on resume')):
                with contextlib.redirect_stdout(io.StringIO()):repeat.main(argv)

    def test_combined_pca64_plan(self):
        with contextlib.redirect_stdout(io.StringIO()):
            count=repeat.main(['--prepared','/absent','--structure','/absent','--smiles','/absent','--out','/absent','--arms','basic','morgan','descriptor','both','--input-components','64','--structure-layout','separate','--output-space','full'])
        self.assertEqual(count,180)

    def test_combined_full_output_repeat(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd,json
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root)
            smiles=root/'smiles.csv'
            pd.DataFrame(dict(compound=[f'compound_{i:02}' for i in range(20)],smiles=['C'*(i+2) for i in range(20)])).to_csv(smiles,index=False)
            argv=['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),'--out',str(root/'repeat'),'--epochs','1','--folds','2','--model-seeds','42','--arms','basic','morgan','descriptor','both','--structure-layout','separate','--input-components','4','--output-space','full','--threads','1','--execute']
            with contextlib.redirect_stdout(io.StringIO()):repeat.main(argv)
            self.assertEqual(len(list((root/'repeat').rglob('complete.json'))),24)
            table=pd.read_csv(root/'repeat/pooled_by_model_seed.csv')
            self.assertEqual(len(table),14)
            self.assertTrue(table.n_targets.eq(40).all())
            paired=pd.read_csv(root/'repeat/paired_rmse_differences.csv')
            self.assertIn('both_minus_morgan_ddpm',set(paired.comparison))
            settings=json.loads((root/'repeat/fold_0/seed_42/run.json').read_text())['settings']
            self.assertEqual(settings['input_components'],4)
            with patch.object(core,'fit_job',side_effect=AssertionError('Must resume')):
                with contextlib.redirect_stdout(io.StringIO()):repeat.main(argv)
            changed=argv.copy();changed[changed.index('--input-components')+1]='3'
            with self.assertRaises(ValueError):
                with contextlib.redirect_stdout(io.StringIO()):repeat.main(changed)
