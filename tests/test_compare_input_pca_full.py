import contextlib,io,tempfile,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_input_pca_full as comparison
import repeated_latent_validation as core

class InputPipelineTests(unittest.TestCase):
    def test_plan_no_io_default_MLP_only(self):
        with contextlib.redirect_stdout(io.StringIO()):
            count=comparison.main(['--prepared','/absent','--structure','/absent','--smiles','/absent','--out','/absent'])
        self.assertEqual(count,60)

    def test_full_pooled_pairing_and_resume_contract(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd,json
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root)
            smiles=root/'smiles.csv';pd.DataFrame({'compound':[f'compound_{i:02}' for i in range(20)],'smiles':['CC']*20}).to_csv(smiles,index=False)
            argv=['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),'--out',str(root/'comparison'),'--folds','2','--model-seeds','42','--epochs','1','--threads','1','--execute']
            with contextlib.redirect_stdout(io.StringIO()):comparison.main(argv)
            folder=root/'comparison'
            paired=pd.read_csv(folder/'paired_by_model_seed.csv')
            self.assertEqual(len(paired),2)
            self.assertEqual(set(paired.arm),{'basic','morgan'})
            self.assertTrue(json.loads((folder/'comparison_status.json').read_text())['all_complete'])
            self.assertEqual(len(list(folder.rglob('complete.json'))),8)
            dims=pd.read_csv(folder/'model_dimensions.csv');self.assertEqual(set(dims.input_space),{'pca','full'})
            self.assertTrue(pd.read_csv(folder/'pooled_metrics.csv').n_targets.eq(40).all())
            with patch.object(core,'fit_job',side_effect=AssertionError('Must resume without training')):
                with contextlib.redirect_stdout(io.StringIO()):comparison.main(argv)
            changed=argv+['--families','ddpm']
            with self.assertRaises(ValueError):
                with contextlib.redirect_stdout(io.StringIO()):comparison.main(changed)
