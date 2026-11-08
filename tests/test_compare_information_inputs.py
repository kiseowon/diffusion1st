import contextlib,io,sys,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_information_inputs as experiment
import meeting_time_structure_pilot as pilot
import repeated_latent_validation as core

class InformationTests(unittest.TestCase):
    def test_default_plan_no_inputs(self):
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(experiment.main(['--prepared','/missing','--structure','/missing','--smiles','/missing','--out','/missing']),60)

    def test_source_ablation_keeps_target_only_and_does_not_mutate(self):
        import pandas as pd,numpy as np,torch
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            a=RepeatedLatentTests().synthetic(Path(d));_,data,ids,meta,Y,_,_=core.load_data(a)
            source=pd.read_csv(a.prepared/'vitro_conditions.csv');source=source[source.condition_id.isin(np.unique(data.source_ids))].reset_index(drop=True)
            pack=pilot.prepare(data,source,pilot.schema_for(source),ids,meta,Y,np.arange(24),components=4,output_space='full')
            original=pack['X'].clone();off=pilot.source_input(pack,'absent');n=off['state']['source_feature_width']
            self.assertEqual(torch.count_nonzero(off['X'][:,:n]).item(),0)
            torch.testing.assert_close(off['X'][:,n:],original[:,n:],rtol=0,atol=0)
            torch.testing.assert_close(pack['X'],original,rtol=0,atol=0)
            self.assertEqual(off['X'].shape,pack['X'].shape)

    def test_complete_pooled_four_conditions_and_resume(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root);smiles=root/'smiles.csv'
            pd.DataFrame({'compound':[f'compound_{i:02}' for i in range(20)],'smiles':['CC']*20}).to_csv(smiles,index=False)
            argv=['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),'--out',str(root/'run'),'--folds','2','--model-seeds','42','--epochs','1','--threads','1','--execute']
            with contextlib.redirect_stdout(io.StringIO()):experiment.main(argv)
            table=pd.read_csv(root/'run/pooled_by_model_seed.csv');self.assertEqual(len(table),6)
            self.assertTrue(table.n_targets.eq(40).all())
            self.assertEqual(len(table[table.model_seed.eq(-1)]),2)
            paired=pd.read_csv(root/'run/paired_rmse.csv');self.assertEqual(len(paired),4)
            self.assertEqual(len(list((root/'run').rglob('complete.json'))),8)
            self.assertTrue(json.loads((root/'run/information_status.json').read_text())['all_complete'])
            with patch.object(core,'fit_job',side_effect=AssertionError('No training on resume')):
                with contextlib.redirect_stdout(io.StringIO()):experiment.main(argv)
            changed=argv.copy();changed[changed.index('--epochs')+1]='2'
            with self.assertRaises(ValueError):
                with contextlib.redirect_stdout(io.StringIO()):experiment.main(changed)
