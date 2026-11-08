import contextlib,io,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import diagnose_unseen_slot_masking as diagnostic
import repeated_latent_validation as core
import meeting_time_structure_pilot as pilot
import aivive_inspired_validation as inspired

class FrozenMaskTests(unittest.TestCase):
    def test_original_reproduction_counterfactual_and_no_training(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd,numpy as np,json
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root)
            source_path=a.prepared/'vitro_conditions.csv';source=pd.read_csv(source_path)
            mapping=core.compound_folds(sorted(source.compound.unique()),42,5)
            special=next(c for c,f in mapping.items() if f==0)
            source.loc[source.compound.eq(special),'unit']='unique_test_unit';source.to_csv(source_path,index=False)
            smiles=root/'smiles.csv';pd.DataFrame({'compound':[f'compound_{i:02}' for i in range(20)],'smiles':['C'*(i+2) for i in range(20)]}).to_csv(smiles,index=False)
            unit=root/'unit'
            with contextlib.redirect_stdout(io.StringIO()):pilot.main(['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),'--out',str(unit),'--families','ddpm','--arms','basic','morgan','descriptor','both','--structure-layout','separate','--input-components','4','--output-space','full','--cache-diffusion-context','--epochs','1','--threads','1','--execute'])
            before={str(p):core.digest(p) for p in unit.rglob('*') if p.is_file()}
            with patch.object(core,'train_epoch',side_effect=AssertionError('No training')),patch.object(inspired,'train',side_effect=AssertionError('No training')):
                with contextlib.redirect_stdout(io.StringIO()):result=diagnostic.diagnose(unit,root/'report',['basic','morgan','descriptor','both'],1)
            self.assertEqual(len(result),4)
            self.assertTrue(result.unseen_weight_max_change_from_seed_initialization.eq(0).all())
            self.assertTrue(result.changed_targets.gt(0).all())
            self.assertTrue(result.reproduction_max_abs_difference.lt(1e-5).all())
            detail=pd.read_csv(root/'report/by_compound.csv')
            unaffected=detail[~detail.changed_input]
            np.testing.assert_allclose(unaffected.original_rmse,unaffected.masked_rmse,rtol=1e-6,atol=1e-6)
            self.assertEqual(before,{str(p):core.digest(p) for p in unit.rglob('*') if p.is_file()})
            # Corrupt only the saved prediction and update its integrity hash to
            # exercise the numerical reproduction guard, not just file checks.
            task=unit/'basic/ddpm';marker=task/'complete.json';record=json.loads(marker.read_text());p=task/record['prediction_path']
            with np.load(p) as z:ids=z['target_ids'].copy();pred=z['prediction'].copy()
            np.savez_compressed(p,target_ids=ids,prediction=pred+1)
            record['sha256'][record['prediction_path']]=core.digest(p);marker.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'failed reproduction'):
                with contextlib.redirect_stdout(io.StringIO()):diagnostic.diagnose(unit,root/'failed',['basic'],1)
            self.assertFalse((root/'failed/basic_diagnostic_predictions.npz').exists())

    def test_numerical_tolerance_requires_explicit_flag_and_all_bounds(self):
        import numpy as np
        saved=np.ones((16,64));saved[0,0]=.001
        truth=np.zeros_like(saved)
        regenerated=saved.copy();regenerated[0,0]+=1.4543533325195312e-5
        strict=diagnostic.reproduction_report(saved,regenerated,truth)
        self.assertFalse(strict['strict_allclose_passed'])
        self.assertFalse(strict['accepted'])
        permitted=diagnostic.reproduction_report(saved,regenerated,truth,True)
        self.assertTrue(permitted['accepted'])
        self.assertLess(permitted['global_rmse_difference'],1e-6)
        too_large=saved.copy();too_large[0,0]+=.001
        self.assertFalse(diagnostic.reproduction_report(saved,too_large,truth,True)['accepted'])
        systematic=saved+2e-5
        self.assertFalse(diagnostic.reproduction_report(saved,systematic,truth,True)['accepted'])
        bad=regenerated.copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):diagnostic.reproduction_report(saved,bad,truth,True)
