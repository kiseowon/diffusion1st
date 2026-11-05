import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import repeated_latent_validation as core
import aivive_inspired_validation as inspired
import meeting_time_structure_pilot as pilot

class FullInputGenerativeTests(unittest.TestCase):
    def test_cached_ddpm_matches_original_sampling_and_calls_encoder_once(self):
        import numpy as np
        import torch
        with patch.dict(core.PROTOCOL,steps=3,samples=2,batch_size=16):
            model,_,_,_=inspired.models('ddpm',17,6,42)
            source=torch.randn(2,10);index=torch.tensor([0,0,1,1])
            features=pilot.SharedSourceFeatures(source,index,torch.randn(4,7))
            pack=dict(X=features)
            with patch.dict(core.PROTOCOL,cache_diffusion_context=False):
                expected=core.predict_scores('ddpm',model,pack,np.arange(4),123)
            calls=[]
            hook=model.context.register_forward_hook(lambda *x:calls.append(1))
            try:
                with patch.dict(core.PROTOCOL,cache_diffusion_context=True):
                    actual=core.predict_scores('ddpm',model,pack,np.arange(4),123)
            finally:hook.remove()
            self.assertEqual(len(calls),1)
            np.testing.assert_allclose(actual,expected,rtol=5e-6,atol=1e-6)

    def test_four_generative_jobs_shared_raw_inputs_and_resume(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd
        import numpy as np
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root)
            smiles=root/'smiles.csv'
            pd.DataFrame(dict(compound=[f'compound_{i:02}' for i in range(20)],smiles=['CC']*20)).to_csv(smiles,index=False)
            argv=['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),
                  '--out',str(root/'generative'),'--input-space','full','--output-space','full',
                  '--families','ddpm','cgan','--arms','basic','morgan','--cache-diffusion-context',
                  '--epochs','1','--threads','1','--execute']
            with contextlib.redirect_stdout(io.StringIO()):pilot.main(argv)
            result=pd.read_csv(root/'generative/results.csv')
            self.assertEqual(len(result),6)
            self.assertEqual(set(result.model),{'ddpm','cgan','zero_response','time_level_mean'})
            self.assertEqual(len(list((root/'generative').rglob('complete.json'))),4)
            self.assertEqual(len(list((root/'generative').rglob('discriminator.pt'))),2)
            for path in (root/'generative').rglob('predictions.npz'):
                with np.load(path,allow_pickle=False) as r:
                    self.assertEqual(r['prediction'].shape[1],6)
                    self.assertTrue(np.isfinite(r['prediction']).all())
            with patch.object(core,'fit_job',side_effect=AssertionError('No retraining on resume')):
                with contextlib.redirect_stdout(io.StringIO()):pilot.main(argv)
