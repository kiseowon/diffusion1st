import unittest,contextlib,io,tempfile,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pairwise_mlp_comparison as pair
import compare_information_inputs as reference
import repeated_latent_validation as core

class PairTests(unittest.TestCase):
    def test_plan_no_io(self):
        with contextlib.redirect_stdout(io.StringIO()):pair.main(['--reference-run','/missing','--out','/missing'])

    def test_inverse_weights_equalize_targets(self):
        import numpy as np
        rows,w=pair.training_weights(np.array([0,0,1,2,2,2]),np.array([0,1]))
        self.assertEqual(list(rows),[0,1,2])
        np.testing.assert_allclose(w[:2].sum(),w[2])
        self.assertEqual(w[3:].sum(),0)
        self.assertAlmostEqual(float(w[rows].mean()),1.)

    def test_pair_predictions_average_before_target_evaluation(self):
        import numpy as np,torch
        pack={'target_index':np.array([0,0,1]),'Z':torch.zeros(2,1),'state':{'output_space':'full','output_scale':1.,'output_mean':np.zeros(1)}}
        class Dummy:pass
        def fake(kind,model,p,rows,seed):return np.array([[1.],[3.],[10.]])[rows]
        old=core.PROTOCOL.copy();core.PROTOCOL['batch_size']=2
        try:
            with patch.object(core,'predict_family_scores',side_effect=fake):pred,counts=pair.predict(Dummy(),pack,np.array([0,1]))
            np.testing.assert_allclose(pred,[[2],[10]])
            np.testing.assert_array_equal(counts,[2,1])
        finally:core.PROTOCOL.clear();core.PROTOCOL.update(old)

    def test_complete_pair_comparison_resume_and_reference_integrity(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd,json,numpy as np,torch,joblib
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root);smiles=root/'smiles.csv'
            pd.DataFrame({'compound':[f'compound_{i:02}' for i in range(20)],'smiles':['CC']*20}).to_csv(smiles,index=False)
            with contextlib.redirect_stdout(io.StringIO()):reference.main(['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),'--out',str(root/'reference'),'--folds','2','--model-seeds','42','--epochs','1','--threads','1','--execute'])
            _,data,ids,meta,Y,_,_=core.load_data(a)
            source=pd.read_csv(a.prepared/'vitro_conditions.csv');source=source[source.condition_id.isin(np.unique(data.source_ids))].reset_index(drop=True)
            state=joblib.load(root/'reference/present/fold_0/seed_42/basic/inner_preprocess.joblib')
            pack=pair.prepare(data,source,ids,meta,Y,state,np.arange(10),'basic')
            changed=Y.copy();changed[10:]+=999
            again=pair.prepare(data,source,ids,meta,changed,state,np.arange(10),'basic')
            torch.testing.assert_close(pack['Z'],again['Z'])
            torch.testing.assert_close(pack['X'][np.arange(3)],again['X'][np.arange(3)])
            argv=['--reference-run',str(root/'reference'),'--out',str(root/'pairs'),'--execute']
            before={str(p):core.digest(p) for p in (root/'reference').rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):pair.main(argv)
            self.assertEqual(before,{str(p):core.digest(p) for p in (root/'reference').rglob('*') if p.is_file()})
            summary=pd.read_csv(root/'pairs/summary.csv');self.assertEqual(len(summary),8)
            metrics=pd.read_csv(root/'pairs/pooled_metrics.csv');self.assertTrue(metrics.n_targets.eq(40).all())
            self.assertEqual(set(metrics.condition)-set(pd.read_csv(root/'reference/pooled_by_model_seed.csv').condition),{'pair_basic','pair_morgan'})
            for f in (root/'pairs').rglob('selection.json'):
                selection=json.loads(f.read_text());self.assertEqual(selection['inner_max_updates'],1)
            with patch.object(pair,'fit',side_effect=AssertionError('No new training on resume')):
                with contextlib.redirect_stdout(io.StringIO()):pair.main(argv)
            # Modifying a saved reference preprocessor invalidates the contract.
            path=root/'reference/present/fold_0/seed_42/basic/inner_preprocess.joblib'
            with path.open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):
                with contextlib.redirect_stdout(io.StringIO()):pair.main(argv)
