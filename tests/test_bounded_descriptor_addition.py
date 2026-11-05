import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import bounded_descriptors as bounded
import repeat_time_structure_pilot as repeat

class BoundedDescriptorTests(unittest.TestCase):
    def test_extreme_heldout_values_bounded_train_only_and_missing(self):
        import numpy as np
        import pandas as pd
        raw=pd.DataFrame(dict(heavy=[1.,10.,1e30,1e300],tiny=[1.,1.+1e-14,1.+2e-14,1e200],
                              nonfinite=[1.,np.nan,3.,4.],constant=[2.,2.,2.,7.]),index=['A','B','C','D'])
        state=bounded.fit(raw,['A','B','C','A'])
        self.assertEqual(state['training_compounds'],['A','B','C'])
        self.assertIn('nonfinite',state['dropped_nonfinite'])
        self.assertIn('constant',state['dropped_constant_or_numerically_degenerate'])
        out,audit=bounded.encode(raw,['D','MISSING'],state,10)
        self.assertTrue(np.isfinite(out).all());self.assertLessEqual(abs(out).max(),1.)
        self.assertGreater(audit['clipped_selected_values'][0],0)
        np.testing.assert_array_equal(out[1],np.zeros(10))

    def test_descriptor_plan_45_jobs_no_io(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(repeat.main(['--prepared','/absent','--structure','/absent','--smiles','/absent',
                                         '--out','/absent','--arms','descriptor']),45)

    def test_descriptor_only_addition_preserves_reference_and_merges(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd
        import json
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root)
            smiles=root/'smiles.csv'
            pd.DataFrame(dict(compound=[f'compound_{i:02}' for i in range(20)],
                              smiles=['C'*(i+2) for i in range(20)])).to_csv(smiles,index=False)
            argv=['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),
                  '--epochs','1','--folds','2','--model-seeds','42','--families','mlp','--threads','1','--execute']
            with contextlib.redirect_stdout(io.StringIO()):repeat.main(argv+['--out',str(root/'old')])
            before=(root/'old/summary.csv').read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                repeat.main(argv+['--out',str(root/'new'),'--arms','descriptor','--reference-run',str(root/'old')])
            self.assertEqual((root/'old/summary.csv').read_bytes(),before)
            combined=pd.read_csv(root/'new/combined_summary.csv')
            self.assertEqual(len(combined),5)
            self.assertEqual(set(combined.arm),{'baseline','basic','morgan','descriptor'})
            self.assertEqual(len(pd.read_csv(root/'new/descriptor_paired_differences.csv')),1)
            unit=json.loads((root/'new/fold_0/seed_42/run.json').read_text())
            self.assertEqual(unit['settings']['descriptor_policy'],'bounded')
            self.assertTrue((root/'new/fold_0/seed_42/descriptor/inner_descriptor_numeric_audit.csv').is_file())
