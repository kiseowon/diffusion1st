import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import meeting_time_structure_pilot as pilot
import repeated_latent_validation as core

class FullProbeTests(unittest.TestCase):
    def test_plan_full_output_no_io(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(pilot.main(['--prepared','/absent','--structure','/absent','--smiles','/absent',
                                         '--out','/absent','--output-space','full']),9)
        settings=json.loads(output.getvalue())['settings']
        self.assertIsNone(settings['output_components'])
        self.assertFalse(settings['output_pca'])

    def test_all_target_dimensions_preserved_and_no_heldout_fit(self):
        import numpy as np
        import pandas as pd
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            a=RepeatedLatentTests().synthetic(Path(d));_,data,ids,meta,Y,_,_=core.load_data(a)
            source=pd.read_csv(a.prepared/'vitro_conditions.csv')
            source=source[source.condition_id.isin(np.unique(data.source_ids))].reset_index(drop=True)
            # More target dimensions than the old output PCA cap.
            Y=np.tile(Y,(1,8));fit=np.arange(24)
            p=pilot.prepare(data,source,pilot.schema_for(source),ids,meta,Y,fit,output_space='full')
            self.assertEqual(p['Z'].shape[1],48)
            self.assertIsNone(p['state']['output_pca'])
            np.testing.assert_allclose(pilot.restore_output(p['Z'][fit].numpy(),p['state']),Y[fit],atol=1e-7)
            changed=Y.copy();changed[24:]+=1000
            q=pilot.prepare(data,source,pilot.schema_for(source),ids,meta,changed,fit,output_space='full')
            np.testing.assert_array_equal(q['Z'],p['Z'])
            np.testing.assert_array_equal(q['state']['output_mean'],p['state']['output_mean'])

    def test_all_models_direct_probe_predictions_and_resume(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd
        import numpy as np
        import torch
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root)
            smiles=root/'smiles.csv'
            pd.DataFrame(dict(compound=[f'compound_{i:02}' for i in range(20)],
                              smiles=['C'*(i+2) for i in range(20)])).to_csv(smiles,index=False)
            argv=['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),
                  '--out',str(root/'full'),'--output-space','full','--epochs','1','--threads','1','--execute']
            with contextlib.redirect_stdout(io.StringIO()):pilot.main(argv)
            self.assertEqual(len(list((root/'full').rglob('complete.json'))),9)
            policy=json.loads((root/'full/target_representation.json').read_text())
            self.assertEqual(policy['inner_dimension'],6)
            self.assertEqual(policy['final_dimension'],6)
            self.assertLess(policy['full_target_roundtrip_max_abs_error'],1e-12)
            for file in (root/'full').rglob('model.pt'):
                weights=torch.load(file,weights_only=True)
                self.assertEqual(weights['head.2.weight'].shape[0],6)
            for file in (root/'full').rglob('predictions.npz'):
                with np.load(file,allow_pickle=False) as pred:self.assertEqual(pred['prediction'].shape[1],6)
            with contextlib.redirect_stdout(io.StringIO()):pilot.main(argv)
