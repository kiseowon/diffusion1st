import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import repeated_latent_validation as run


class RepeatedLatentTests(unittest.TestCase):
    def synthetic(self, root):
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(82)
        prepared = root / 'prepared'; prepared.mkdir()
        source, target, sc, tc, pairs, structure = [], [], [], [], [], []
        for i in range(20):
            name = f'compound_{i:02}'
            base_s = rng.normal(size=6); base_t = rng.normal(size=6)
            response = rng.normal(scale=.2, size=6)
            sids, tids = [], []
            for level, dose in [('Control', 0.), ('Low', 1.), ('High', 2.)]:
                s = len(sc); t = len(tc)
                sc.append(dict(condition_id=s, compound=name, dose=dose, unit='unit', time_h=24., level=level))
                tc.append(dict(condition_id=t, compound=name, dose=dose, unit='unit', time_h=24., level=level))
                source.append(base_s + dose * response)
                target.append(base_t + dose * (.6 * response + .05))
                sids.append(s); tids.append(t)
            for s in sids[1:]:
                for t in tids:
                    record = dict(compound=name, condition_id_source=s, condition_id_target=t)
                    for suffix, row in [('source', sc[s]), ('target', tc[t])]:
                        record.update({k + '_' + suffix: row[k] for k in ['dose', 'unit', 'time_h', 'level']})
                    pairs.append(record)
            structure.append(dict(compound=name, bit0=i % 2, bit1=(i // 2) % 2))
        np.save(prepared / 'vitro_expression.npy', np.asarray(source, dtype=np.float32))
        np.save(prepared / 'vivo_expression.npy', np.asarray(target, dtype=np.float32))
        pd.DataFrame(sc).to_csv(prepared / 'vitro_conditions.csv', index=False)
        pd.DataFrame(tc).to_csv(prepared / 'vivo_conditions.csv', index=False)
        pd.DataFrame(pairs).to_csv(prepared / 'pairs.csv.gz', index=False)
        pd.DataFrame(dict(probe=[f'p{i}' for i in range(6)])).to_csv(prepared / 'probes.csv', index=False)
        st = root / 'structure.csv'; pd.DataFrame(structure).to_csv(st, index=False)
        return run.parser().parse_args(['--prepared', str(prepared), '--structure', str(st),
            '--out', str(root / 'output'), '--folds', '2', '--model-seeds', '42',
            '--threads', '1', '--bootstrap-samples', '30', '--execute'])

    def test_plan_never_reads_inputs_or_trains(self):
        with tempfile.TemporaryDirectory() as d, patch.object(run, 'execute') as execute:
            out = Path(d) / 'not_created'
            with contextlib.redirect_stdout(io.StringIO()):
                jobs = run.main(['--prepared', '/missing', '--structure', '/missing.csv', '--out', str(out)])
            self.assertEqual(len(jobs), 30)
            execute.assert_not_called()
            self.assertFalse(out.exists())

    def test_folds_cover_each_compound_once_and_inner_is_disjoint(self):
        names = [f'c{i}' for i in range(140)]
        mapping = run.compound_folds(names, 42, 5)
        self.assertEqual(mapping, run.compound_folds(names, 42, 5))
        self.assertNotEqual(mapping, run.compound_folds(names, 43, 5))
        for f in range(5):
            test = {n for n, k in mapping.items() if k == f}
            train, val = run.inner_compounds(mapping, f, 42)
            self.assertEqual((len(train), len(val), len(test)), (89, 23, 28))
            self.assertFalse(train & val or train & test or val & test)
            self.assertEqual(train | val | test, set(names))

    def test_preprocessing_training_only_and_no_heldout_training_targets(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            args = self.synthetic(Path(d))
            first, data, ids, meta, Y, _, _ = run.load_data(args)
            fit = np.arange(16)
            baseline, _ = run.baseline_stage(meta, Y, fit, 10)
            pack = run.prepare_stage(first, data, ids, meta, Y, fit, baseline)
            changed_y = Y.copy(); changed_y[16:] += 1000
            changed_data = __import__('copy').deepcopy(data)
            compound_names = set(meta.iloc[16:].compound)
            changed_source_ids = np.unique(data.source_ids[data.pairs.compound.isin(compound_names)])
            changed_data.source[changed_source_ids] += 500
            b2, _ = run.baseline_stage(meta, changed_y, fit, 10)
            p2 = run.prepare_stage(first, changed_data, ids, meta, changed_y, fit, b2)
            np.testing.assert_allclose(pack['X'][fit].numpy(), p2['X'][fit].numpy())
            np.testing.assert_allclose(pack['Z'][fit].numpy(), p2['Z'][fit].numpy())
            self.assertTrue((pack['Z'][16:] == 0).all())
            self.assertEqual(pack['state']['output_scale'], p2['state']['output_scale'])

    def test_both_prediction_paths_are_finite_and_deterministic(self):
        import torch
        import numpy as np
        with patch.dict(run.PROTOCOL, steps=3, samples=2):
            model, opt = run.make_model(4, 3, 42)
            pack = dict(X=torch.randn(5, 4), Z=torch.randn(5, 3))
            for kind in ['mlp', 'ddpm']:
                run.train_epoch(kind, model, opt, pack, np.arange(4))
                pred = run.predict_scores(kind, model, pack, np.arange(5), 99)
                self.assertEqual(pred.shape, (5, 3))
                self.assertTrue(np.isfinite(pred).all())
                np.testing.assert_array_equal(pred, run.predict_scores(kind, model, pack, np.arange(5), 99))

    def test_three_family_plan_and_cgan_updates_both_networks(self):
        import torch
        import numpy as np
        with contextlib.redirect_stdout(io.StringIO()):
            planned = run.main(['--prepared', '/missing', '--structure', '/missing.csv',
                                '--out', '/missing_output', '--families', 'mlp', 'cgan', 'ddpm'])
        self.assertEqual(len(planned), 45)
        with patch.dict(run.PROTOCOL, samples=2):
            model, opt, disc, dopt = run.make_family_model('cgan', 4, 3, 42)
            pack = dict(X=torch.randn(5, 4), Z=torch.randn(5, 3))
            before_g = [p.detach().clone() for p in model.parameters()]
            before_d = [p.detach().clone() for p in disc.parameters()]
            for _ in range(2):
                run.train_family_epoch('cgan', model, opt, disc, dopt, pack, np.arange(4))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_g, model.parameters())))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_d, disc.parameters())))
            pred = run.predict_family_scores('cgan', model, pack, np.arange(5), 99)
            self.assertEqual(pred.shape, (5, 3))
            self.assertTrue(np.isfinite(pred).all())
            np.testing.assert_array_equal(pred, run.predict_family_scores('cgan', model, pack, np.arange(5), 99))

    def test_time_level_baseline_compound_balance(self):
        import pandas as pd
        import numpy as np
        meta = pd.DataFrame(dict(compound=['A', 'A', 'B', 'T'], time_h=[24.] * 4, level=['High'] * 4))
        y = np.array([[0.], [2.], [5.], [1000.]])
        pred, counts = run.time_level_mean(meta, y, np.array([0, 1, 2]), np.array([3]))
        self.assertEqual(pred.item(), 3.)
        self.assertEqual(counts[0]['compounds'], 2)

    def test_synthetic_end_to_end_resume_contract_and_artifact_checks(self):
        import pandas as pd
        reduced = dict(epochs=1, validation_epochs=[1], steps=3, samples=2,
                       input_components=4, output_components=3, baseline_alphas=[10.], strengths=[0., 1.])
        with tempfile.TemporaryDirectory() as d, patch.dict(run.PROTOCOL, **reduced):
            args = self.synthetic(Path(d))
            with contextlib.redirect_stdout(io.StringIO()):
                run.execute(args)
            done = json.loads((args.out / 'completed.json').read_text())
            self.assertEqual(done['completed_jobs'], 4)
            all_rows = pd.read_csv(args.out / 'oof_condition_metrics.csv')
            baselines = all_rows.loc[all_rows.model.eq('time_level_mean')]
            self.assertEqual(len(baselines), 40)
            self.assertFalse(baselines.duplicated(['split_seed', 'condition_id']).any())
            self.assertTrue((args.out / 'compound_bootstrap.csv').is_file())
            with patch.object(run, 'fit_job', side_effect=AssertionError('resume must skip fits')):
                with contextlib.redirect_stdout(io.StringIO()):
                    run.execute(args)
            args.model_seeds = [43]
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                run.execute(args)
            args.model_seeds = [42]
            prediction = next(args.out.glob('split_*/fold_*/mlp_seed*/attempt_*/predictions.npz'))
            prediction.write_bytes(prediction.read_bytes() + b'tamper')
            with self.assertRaisesRegex(ValueError, 'changed'):
                with contextlib.redirect_stdout(io.StringIO()):
                    run.execute(args)

    def test_nested_seed_aggregation_does_not_ensemble_predictions(self):
        import pandas as pd
        # Same target, two model seeds. Squared errors average first.
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); records = []
            for model, errors in [('mlp', [1., 9.]), ('ddpm', [1., 1.])]:
                for seed, err in enumerate(errors):
                    p = out / f'{model}_{seed}.csv'
                    pd.DataFrame([dict(condition_id=1, compound='A', response_mse=err)]).to_csv(p, index=False)
                    records.append(dict(job=dict(split_seed=42, fold=0, model_seed=seed, model=model), metrics_path=p.name))
            baseline = [dict(split_seed=42, fold=0, condition_id=1, compound='A', target_conditions_ridge_mse=4., time_level_mean_mse=4., zero_response_mse=9.)]
            run.aggregate(out, records, baseline, 0)
            summary = pd.read_csv(out / 'summary.csv').set_index('model')
            self.assertAlmostEqual(summary.loc['mlp', 'pooled_rmse'], 5 ** .5)
            self.assertEqual(summary.loc['ddpm', 'pooled_rmse'], 1.)


if __name__ == '__main__':
    unittest.main()
