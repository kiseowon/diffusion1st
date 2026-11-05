import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import meeting_time_structure_pilot as pilot
import repeated_latent_validation as core

class MeetingPilotTests(unittest.TestCase):
    def test_plan_no_io(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.main(['--prepared','/missing','--structure','/missing','--smiles','/missing','--out','/missing']),9)

    def test_slots_masks_and_heldout_truth_not_used(self):
        import numpy as np
        import pandas as pd
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            a=RepeatedLatentTests().synthetic(Path(d));_,data,ids,meta,Y,_,_=core.load_data(a)
            source=pd.read_csv(a.prepared/'vitro_conditions.csv')
            source=source[source.condition_id.isin(np.unique(data.source_ids))].reset_index(drop=True)
            slots=pilot.schema_for(source);fit=np.arange(24)
            p=pilot.prepare(data,source,slots,ids,meta,Y,fit)
            changed=Y.copy();changed[24:]+=123
            q=pilot.prepare(data,source,slots,ids,meta,changed,fit)
            np.testing.assert_array_equal(p['X'],q['X']);np.testing.assert_array_equal(p['Z'],q['Z'])
            self.assertEqual(len(slots),2)
            per_slot=(p['X'].shape[1]-p['state']['target_condition_encoder'].transform(
                pd.DataFrame(dict(dose=np.log1p(meta.dose),time=meta.time_h,unit=meta.unit,level=meta.level))).shape[1])//2
            self.assertEqual(float(p['X'][0,per_slot-1]),1.)
            # Remove one observed source condition; it becomes a masked empty slot, not an average.
            source2=source.drop(index=0).reset_index(drop=True)
            r=pilot.prepare(data,source2,slots,ids,meta,Y,fit)
            removed=source.iloc[0]
            slot=slots.index((float(removed.time_h),str(removed.level),str(removed.unit),0))
            self.assertEqual(float(r['X'][0,(slot+1)*per_slot-1]),0.)
            self.assertEqual(float(r['X'][0,(2-slot)*per_slot-1]),1.)

    def test_nine_synthetic_jobs_and_resume(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit unavailable')
        import pandas as pd
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=RepeatedLatentTests().synthetic(root)
            smiles=root/'smiles.csv'
            pd.DataFrame(dict(compound=[f'compound_{i:02}' for i in range(20)],
                              smiles=['C'*(i+2) for i in range(20)])).to_csv(smiles,index=False)
            argv=['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),
                  '--out',str(root/'pilot'),'--epochs','1','--threads','1','--execute']
            with contextlib.redirect_stdout(io.StringIO()):pilot.main(argv)
            self.assertEqual(len(pd.read_csv(root/'pilot/results.csv')),11)
            self.assertEqual(len(list((root/'pilot').rglob('complete.json'))),9)
            with patch.object(core,'fit_job',side_effect=AssertionError('Must resume')):
                with contextlib.redirect_stdout(io.StringIO()):pilot.main(argv)
