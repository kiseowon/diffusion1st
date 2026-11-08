import unittest,contextlib,io,tempfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import inspect_saved_input_ranges as diagnostic
import meeting_time_structure_pilot as pilot
import repeated_latent_validation as core

class InputDiagnosticTests(unittest.TestCase):
    def test_saved_transforms_only_and_input_integrity(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd
        from unittest.mock import patch
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root)
            smiles=root/'smiles.csv';pd.DataFrame({'compound':[f'compound_{i:02}' for i in range(20)],'smiles':['CC']*20}).to_csv(smiles,index=False)
            unit=root/'unit'
            with contextlib.redirect_stdout(io.StringIO()):pilot.main(['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),'--out',str(unit),'--families','mlp','--arms','basic','--input-components','4','--output-space','full','--epochs','1','--threads','1','--execute'])
            before={str(p):core.digest(p) for p in unit.rglob('*') if p.is_file()}
            with patch.object(core,'fit_job',side_effect=AssertionError('No training')),patch.object(core,'predict_family_scores',side_effect=AssertionError('No predictions')):
                with contextlib.redirect_stdout(io.StringIO()):summary=diagnostic.inspect(unit,root/'report')
            self.assertTrue(summary.partition.eq('test').any())
            self.assertEqual(before,{str(p):core.digest(p) for p in unit.rglob('*') if p.is_file()})
            with self.assertRaises(ValueError):diagnostic.inspect(unit,root/'report')
            with (a.prepared/'vitro_conditions.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):diagnostic.inspect(unit,root/'report2')
