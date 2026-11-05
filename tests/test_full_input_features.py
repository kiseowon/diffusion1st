import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import meeting_time_structure_pilot as pilot
import aivive_inspired_validation as inspired
import repeated_latent_validation as core

class FullInputTests(unittest.TestCase):
    def test_full_input_plan_no_io(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(pilot.main(['--prepared','/absent','--structure','/absent','--smiles','/absent',
                '--out','/absent','--input-space','full','--output-space','full','--arms','basic','morgan','--families','mlp']),2)
        settings=json.loads(output.getvalue())['settings']
        self.assertIsNone(settings['input_components']);self.assertFalse(settings['pca'])

    def test_source_shared_per_compound_preserves_each_probe_and_no_heldout_fit(self):
        import numpy as np
        import pandas as pd
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            a=RepeatedLatentTests().synthetic(Path(d));_,data,ids,meta,Y,_,_=core.load_data(a)
            source=pd.read_csv(a.prepared/'vitro_conditions.csv')
            source=source[source.condition_id.isin(np.unique(data.source_ids))].reset_index(drop=True)
            fit=np.arange(24);slots=pilot.schema_for(source)
            pack=pilot.prepare(data,source,slots,ids,meta,Y,fit,input_space='full',output_space='full')
            self.assertIsNone(pack['state']['source_pca'])
            self.assertEqual(pack['state']['source_expression_dimension'],6)
            self.assertIsInstance(pack['X'],pilot.SharedSourceFeatures)
            self.assertEqual(pack['X'].source.shape[0],20)
            self.assertEqual(pack['X'].shape[0],40)
            row=source.iloc[0];slot=slots.index((float(row.time_h),str(row.level),str(row.unit),0))
            target_width=pack['X'].target.shape[1]
            slotwidth=(pack['X'].shape[1]-target_width)//len(slots)
            expected=pack['state']['source_scaler'].transform(data.source[[int(row.condition_id)]])[0]
            np.testing.assert_allclose(pack['X'][0,slot*slotwidth:slot*slotwidth+6].numpy(),expected,atol=1e-7)
            changed=Y.copy();changed[24:]+=1000
            other=pilot.prepare(data,source,slots,ids,meta,changed,fit,input_space='full',output_space='full')
            np.testing.assert_array_equal(pack['X'][[0,5,29]],other['X'][[0,5,29]])

    def test_two_mlp_jobs_output_dimension_and_parameter_report(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd
        import torch
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root)
            smiles=root/'smiles.csv'
            pd.DataFrame(dict(compound=[f'compound_{i:02}' for i in range(20)],smiles=['CC']*20)).to_csv(smiles,index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                pilot.main(['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),
                    '--out',str(root/'raw'),'--input-space','full','--output-space','full',
                    '--arms','basic','morgan','--families','mlp','--epochs','1','--threads','1','--execute'])
            self.assertEqual(len(list((root/'raw').rglob('complete.json'))),2)
            self.assertEqual(len(pd.read_csv(root/'raw/results.csv')),4)
            counts=[]
            for arm in ['basic','morgan']:
                report=json.loads((root/'raw'/arm/'model_dimensions.json').read_text())['final']
                weights=torch.load(next((root/'raw'/arm).rglob('model.pt')),weights_only=True)
                actual=sum(v.numel() for v in weights.values())
                self.assertEqual(actual,report['mlp_parameters']);counts.append(actual)
            self.assertEqual(counts[0],counts[1])
