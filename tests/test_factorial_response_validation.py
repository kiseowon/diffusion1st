import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import factorial_response_validation as factorial
import repeated_latent_validation as core


class FactorialTests(unittest.TestCase):
    def test_plan_without_inputs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(factorial.main(['--prepared','/absent','--structure','/absent.csv',
                                            '--kegg','/absent','--out','/absent']),180)

    def test_all_arms_all_families_and_probe_order_guard(self):
        import numpy as np
        import pandas as pd
        from test_repeated_latent_validation import RepeatedLatentTests
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = RepeatedLatentTests().synthetic(root)
            kegg = root/'kegg'; kegg.mkdir()
            probes = pd.read_csv(args.prepared/'probes.csv').probe.astype(str).tolist()
            np.save(kegg/'projection.npy', np.ones((len(probes),2))/len(probes))
            core.save_json(kegg/'provenance.json',dict(probe_order=probes[::-1]))
            with self.assertRaisesRegex(ValueError,'probe_order'):
                factorial.read_projection(args.prepared,kegg)
            core.save_json(kegg/'provenance.json',dict(probe_order=probes))
            reduced=dict(epochs=1,validation_epochs=[1],steps=3,samples=2,input_components=4,output_components=3)
            with patch.dict(core.PROTOCOL, **reduced), patch.object(core,'baseline_stage',side_effect=AssertionError('No Ridge')):
                with contextlib.redirect_stdout(io.StringIO()):
                    factorial.main(['--prepared',str(args.prepared),'--structure',str(args.structure),
                                    '--kegg',str(kegg),'--out',str(root/'factorial'),
                                    '--model-seeds','42','--folds','2','--threads','1','--execute'])
            summary = pd.read_csv(root/'factorial/factorial_summary.csv')
            self.assertEqual(len(summary),20) # 12 models + two baselines per arm
            for arm in factorial.ARMS:
                rows=pd.read_csv(root/'factorial'/arm/'results.csv')
                self.assertEqual(len(rows),6)
                self.assertTrue(rows.strength.eq(1).all())
                contract=json.loads((root/'factorial'/arm/'run.json').read_text())
                self.assertEqual(contract['protocol']['factorial_arm'],arm)
                self.assertIn(str((kegg/'projection.npy').resolve()),contract['input_sha256'])
                self.assertIn(str(Path(factorial.__file__).resolve()),contract['source_sha256'])
