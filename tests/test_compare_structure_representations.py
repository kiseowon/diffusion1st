import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_structure_representations as study
import aivive_inspired_validation as inspired
import repeated_latent_validation as core

class RepresentationTests(unittest.TestCase):
    def test_plan_no_input_io(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(study.main(['--prepared','/absent','--structure','/absent',
                                        '--smiles','/absent','--out','/absent']),45)

    def test_train_only_selection_unique_compounds_and_missing_encoding(self):
        import numpy as np
        import pandas as pd
        raw=pd.DataFrame(dict(a=[0.,1.,2.,100.],b=[0.,2.,4.,-99.],constant=[1.,1.,1.,5.],
                              bad=[np.nan,1.,2.,3.],test_missing=[2.,0.,1.,np.nan]),
                         index=['A','B','C','D'])
        state=study.fit_descriptors(raw,['A','B','C','A'])
        self.assertEqual(state['training_compounds'],['A','B','C'])
        self.assertNotIn('b',state['selected'])
        self.assertIn('bad',state['dropped_nonfinite'])
        self.assertIn('constant',state['dropped_constant'])
        self.assertIn('test_missing',state['selected'])
        values,count=study.encode_descriptors(raw,['D','UNKNOWN'],state,10)
        self.assertEqual(count[0],1)
        self.assertEqual(values[0,state['selected'].index('test_missing')],0)
        self.assertEqual(values[0,-1],1)
        np.testing.assert_array_equal(values[1],np.zeros(10))
        self.assertEqual(state['scaler'].mean_[state['selected'].index('a')],1.)

    def test_real_rdkit_and_three_arm_synthetic_fit(self):
        try: import rdkit
        except ImportError: self.skipTest('RDKit not installed')
        import pandas as pd
        import torch
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root)
            smiles=root/'smiles.csv'
            pd.DataFrame({'compound':[f'compound_{i:02}' for i in range(20)],
                          'smiles':['C'*(i+2) for i in range(20)]}).to_csv(smiles,index=False)
            raw,audit,version=study.calculate_descriptors(smiles)
            self.assertIn('MolWt',raw.columns)
            self.assertNotIn('RadiusOfGyration',raw.columns)
            self.assertTrue(audit.parsed.all())
            with patch.dict(inspired.SETTINGS,epochs=1,validation_epochs=[1]),patch.dict(core.PROTOCOL,steps=3,samples=2):
                with contextlib.redirect_stdout(io.StringIO()):
                    study.main(['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),
                                '--out',str(root/'run'),'--folds','2','--model-seeds','42','--threads','1',
                                '--bootstrap-samples','0','--execute'])
            summary=pd.read_csv(root/'run/representation_summary.csv')
            self.assertEqual(len(summary),9)
            self.assertEqual(set(summary.arm),{'basic','descriptor','morgan'})
            sizes=[]
            for arm in ['basic','descriptor','morgan']:
                checkpoint=next((root/'run'/arm).rglob('model.pt'))
                weights=torch.load(checkpoint,weights_only=True)
                sizes.append(sum(w.numel() for w in weights.values()))
            self.assertEqual(len(set(sizes)),1)
            self.assertFalse(list((root/'run').rglob('ridge.joblib')))
