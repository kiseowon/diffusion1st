import unittest,contextlib,io,sys,tempfile,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import diagnose_toy_fit as diagnostic
import fixed_condition_toy as toy
import repeated_latent_validation as core

class ToyFitDiagnosticsTests(unittest.TestCase):
    def test_frozen_final_train_test_and_denoising(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd
        from test_fixed_condition_toy import ToyTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a,smiles,*_=ToyTests().make(root)
            argv=['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),'--out',str(root/'toy'),'--input-components','4','--epochs','1','--model-seeds','42','--threads','1','--execute']
            with contextlib.redirect_stdout(io.StringIO()):toy.main(argv)
            before={str(p):core.digest(p) for p in (root/'toy').rglob('*') if p.is_file()}
            with patch.object(core,'fit_job',side_effect=AssertionError('No training')),patch.object(toy,'prepare',side_effect=AssertionError('No refitting')):
                with contextlib.redirect_stdout(io.StringIO()):metrics=diagnostic.diagnose(root/'toy',root/'report')
            self.assertEqual(len(metrics),18)
            self.assertEqual(set(metrics.phase),{'final_training','test'})
            self.assertTrue(metrics[metrics.phase.eq('final_training')].n_targets.eq(16).all())
            self.assertTrue(metrics[metrics.phase.eq('test')].n_targets.eq(4).all())
            denoise=pd.read_csv(root/'report/ddpm_training_denoising.csv')
            self.assertEqual(len(denoise),15)
            self.assertTrue(denoise.uses_true_training_target_in_noisy_input.all())
            for path in (root/'report').glob('*_reproduction.json'):
                self.assertTrue(json.loads(path.read_text())['accepted'])
            self.assertEqual(before,{str(p):core.digest(p) for p in (root/'toy').rglob('*') if p.is_file()})
            self.assertFalse(json.loads((root/'report/provenance.json').read_text())['training'])
            with self.assertRaises(ValueError):diagnostic.diagnose(root/'toy',root/'report')
