import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import aivive_inspired_validation as study
import repeated_latent_validation as core

class AiviveInspiredTests(unittest.TestCase):
    def test_plan_has_no_io(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(study.main(['--prepared','/absent','--structure','/absent','--out','/absent']),90)

    def test_no_pca_train_only_scalers_and_all_families(self):
        import numpy as np
        import pandas as pd
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            args=RepeatedLatentTests().synthetic(Path(d))
            first,data,ids,meta,Y,_,_=core.load_data(args)
            fit=np.arange(16)
            pack=study.prepare(first,data,ids,meta,Y,fit,np.zeros_like(Y))
            self.assertEqual(pack['Z'].shape[1],Y.shape[1])
            np.testing.assert_allclose(pack['state']['output_scaler'].data_min_,Y[fit].min(0))
            np.testing.assert_allclose(study.restore(pack['Z'][fit].numpy(),pack['state']),Y[fit],atol=1e-7)
            self.assertFalse(pack['state']['pca'])
            reduced=dict(epochs=1,validation_epochs=[1])
            with patch.dict(study.SETTINGS,**reduced),patch.dict(core.PROTOCOL,steps=3,samples=2):
                with contextlib.redirect_stdout(io.StringIO()):
                    study.main(['--prepared',str(args.prepared),'--structure',str(args.structure),
                                '--out',str(Path(d)/'run'),'--folds','2','--model-seeds','42',
                                '--threads','1','--bootstrap-samples','0','--execute'])
            summary=pd.read_csv(Path(d)/'run/structure_summary.csv')
            self.assertEqual(len(summary),10)
            self.assertEqual(set(summary.model),{'mlp','cgan','ddpm','time_level_mean','zero_response'})
            self.assertFalse(list((Path(d)/'run').rglob('ridge.joblib')))
