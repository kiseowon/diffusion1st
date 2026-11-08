import contextlib,io,tempfile,sys,unittest,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import random_output_subset_toy as subset
import fixed_condition_toy as toy
import repeated_latent_validation as core

class OutputSubsetTests(unittest.TestCase):
    def test_plan_never_reads_reference_or_creates_output(self):
        with tempfile.TemporaryDirectory() as d,patch.object(subset,'verify_reference',side_effect=AssertionError('No reference reads')) as verify:
            out=Path(d)/'planned'
            with contextlib.redirect_stdout(io.StringIO()):subset.main(['--reference-run','/missing','--out',str(out)])
            verify.assert_not_called();self.assertFalse(out.exists())

    def test_selection_fixed_unique_and_in_original_order(self):
        import numpy as np
        ix=subset.selected_indices(3453,256,20261109)
        np.testing.assert_array_equal(ix,subset.selected_indices(3453,256,20261109))
        self.assertEqual(len(ix),256);self.assertTrue(np.all(np.diff(ix)>0))
        self.assertFalse(np.array_equal(ix,subset.selected_indices(3453,256,20261110)))
        with self.assertRaises(ValueError):subset.selected_indices(6,7,42)

    def test_subset_target_transform_keeps_reference_scale_and_heldout_zero(self):
        import numpy as np,torch
        Y=np.arange(30,dtype=float).reshape(5,6)
        state={'output_mean':np.arange(6,dtype=float),'output_scale':7.,'output_space':'full'}
        X=torch.ones(5,4);fit=np.array([0,1,2]);indices=np.array([1,4])
        a=subset.subset_pack(X,Y,state,fit,indices)
        changed=Y.copy();changed[3:]+=999
        b=subset.subset_pack(X,changed,state,fit,indices)
        torch.testing.assert_close(a['Z'],b['Z'])
        self.assertEqual(a['state']['output_scale'],7.)
        self.assertTrue(a['X'] is X)
        np.testing.assert_array_equal(state['output_mean'],np.arange(6))
        np.testing.assert_allclose(a['Z'][fit].numpy()*7+state['output_mean'][indices],Y[fit][:,indices],rtol=1e-6)

    def test_nine_subset_models_same_probe_reference_and_resume(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd,numpy as np
        from test_fixed_condition_toy import ToyTests
        from diagnose_toy_fit import reconstruct
        from evaluate_repeated_predictions import score
        import joblib
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a,smiles,*_=ToyTests().make(root);run=root/'toy'
            with contextlib.redirect_stdout(io.StringIO()):toy.main(['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),'--out',str(run),'--input-components','4','--epochs','1','--model-seeds','42','--threads','1','--execute'])
            before={str(p):core.digest(p) for p in run.rglob('*') if p.is_file()}
            argv=['--reference-run',str(run),'--out',str(root/'subset'),'--n-probes','3','--probe-seed','17','--execute']
            epochs=[];original=core.fit_job
            def check_fit(job,*args,**kwargs):
                self.assertEqual(core.PROTOCOL['epochs'],1)
                self.assertEqual(core.PROTOCOL['validation_epochs'],[1]);epochs.append(1)
                return original(job,*args,**kwargs)
            with patch.object(core,'fit_job',side_effect=check_fit):
                with contextlib.redirect_stdout(io.StringIO()):subset.main(argv)
            self.assertEqual(len(epochs),9)
            table=pd.read_csv(root/'subset/by_seed.csv');self.assertEqual(len(table),20)
            self.assertTrue(table.n_probes.eq(3).all());self.assertTrue(table.n_targets.eq(4).all())
            self.assertEqual(len(table[table.output_training.eq('baseline')]),2)
            pairs=pd.read_csv(root/'subset/paired_by_seed.csv');self.assertEqual(len(pairs),9)
            indices=pd.read_csv(root/'subset/selected_probes.csv').original_probe_index.to_numpy(int)
            state=joblib.load(run/'basic/final_preprocess.joblib');cohort=pd.read_csv(run/'cohort.csv')
            _,truth=reconstruct(run,state,cohort,a.prepared,a.structure);test=np.flatnonzero(pd.read_csv(run/'compound_split.csv').stage.eq('test'))
            task=run/'basic/seed_42/mlp';marker=json.loads((task/'complete.json').read_text())
            with np.load(task/marker['prediction_path']) as saved:pred=saved['prediction'][:,indices]
            expected=score(truth[test][:,indices],pred,[.5,1.])[0]['rmse']
            row=table[table.output_training.eq('full_model_on_same_selected_probes')&table.arm.eq('basic')&table.model.eq('mlp')].iloc[0]
            self.assertAlmostEqual(row.rmse,expected)
            self.assertEqual(before,{str(p):core.digest(p) for p in run.rglob('*') if p.is_file()})
            with patch.object(core,'fit_job',side_effect=AssertionError('Must resume')):
                with contextlib.redirect_stdout(io.StringIO()):subset.main(argv)
            changed=argv.copy();changed[changed.index('--probe-seed')+1]='18'
            with self.assertRaises(ValueError):
                with contextlib.redirect_stdout(io.StringIO()):subset.main(changed)
