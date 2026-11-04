"""Plan or run compound-fold repeats of a fixed latent MLP/x0-DDPM protocol.

Default is plan-only: no inputs are read and no outputs are created.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import uuid

SNAPSHOT = (Path(__file__).resolve().parents[1] / 'results' /
            'controlled_latent_mlp_ddpm_20261104_044442_34d1be' /
            'dependencies/parent_lowrank/first_structure_mlp.py')
PROTOCOL = dict(input_components=64, output_components=32, rank_tolerance=1e-8,
                epochs=100, validation_epochs=[1, 25, 50, 75, 100],
                learning_rate=.0003, weight_decay=.01, batch_size=64,
                gradient_clip=1., steps=30, samples=16,
                strengths=[0., .1, .25, .5, 1.], baseline_alphas=[10., 100., 1000.],
                structure=True, prediction='x0', schedule='cosine', device='cpu',
                residual='in_sample_training_ridge_not_oof',
                input_aggregation='source_features_averaged_per_target',
                target='vivo_treatment_minus_matched_control')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def jobs(args):
    return [dict(split_seed=s, fold=f, model_seed=m, model=k)
            for s in args.split_seeds for f in range(args.folds)
            for m in args.model_seeds for k in ['mlp', 'ddpm']]


def compound_folds(names, seed, folds):
    import numpy as np
    names = np.random.default_rng(seed).permutation(sorted(names))
    if len(set(names)) != len(names) or len(names) < folds * 3:
        raise ValueError('Need unique compounds and at least three compounds per fold')
    return {str(name): int(i % folds) for i, name in enumerate(names)}


def inner_compounds(mapping, fold, seed):
    import numpy as np
    names = sorted(n for n, f in mapping.items() if f != fold)
    # A separate deterministic seed for each fold; shared across both model families/seeds.
    names = np.random.default_rng(np.random.SeedSequence([seed, fold, 741])).permutation(names)
    nval = max(1, int(np.ceil(.2 * len(names))))
    return set(names[nval:]), set(names[:nval])


def protect_output(out, inputs):
    out = out.resolve()
    for p in inputs:
        p = p.resolve()
        if p == out or p in out.parents or out in p.parents:
            raise ValueError('Output must be separate from input paths')


def contract_folder(out, contract):
    if out.exists() and any(out.iterdir()):
        if not (out / 'run.json').is_file() or json.loads((out / 'run.json').read_text()) != contract:
            raise ValueError('Resume contract differs; use a new output folder')
    else:
        out.mkdir(parents=True, exist_ok=True)
        save_json(out / 'run.json', contract)


def read_completion(folder, job):
    marker = folder / 'complete.json'
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    if record['job'] != job:
        raise ValueError('Completed job identity mismatch')
    for relative, expected in record['sha256'].items():
        p = folder / relative
        if not p.is_file() or digest(p) != expected:
            raise ValueError(f'Completed artifact missing or changed: {p}')
    return record


def load_data(args):
    import numpy as np
    import pandas as pd
    spec = importlib.util.spec_from_file_location('_ivive_loader_snapshot', SNAPSHOT)
    first = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(first)
    data = first.Data(args)
    source_table = pd.read_csv(args.prepared / 'vitro_conditions.csv')
    def controls(table):
        c = table.loc[table.level.eq('Control')]
        if c.duplicated(['compound', 'unit', 'time_h']).any():
            raise ValueError('Ambiguous controls')
        return c.set_index(['compound', 'unit', 'time_h']).condition_id.to_dict()
    target_control = controls(data.vo)
    source_control = controls(source_table)
    ids = np.unique(data.target_ids[~data.pairs.level_target.eq('Control').to_numpy()])
    meta = data.vo.iloc[ids].reset_index(drop=True)
    Y, target_audit = [], []
    for cid in ids:
        row = data.vo.iloc[cid]
        key = (row.compound, row.unit, row.time_h)
        if key not in target_control:
            raise ValueError(f'Target control missing: {key}')
        control = int(target_control[key])
        Y.append(data.target[cid] - data.target[control])
        target_audit.append(dict(condition_id=int(cid), control_condition_id=control))
    source_delta = np.zeros_like(data.source, dtype=np.float32)
    source_audit = []
    for sid in np.unique(data.source_ids):
        row = source_table.iloc[sid]
        key = (row.compound, row.unit, row.time_h)
        if key not in source_control:
            raise ValueError(f'Source control missing: {key}')
        control = int(source_control[key])
        source_delta[sid] = data.source[sid] - data.source[control]
        source_audit.append(dict(condition_id=int(sid), control_condition_id=control))
    data.source = source_delta
    return first, data, ids, meta, np.asarray(Y, dtype=np.float64), target_audit, source_audit


def baseline_stage(meta, Y, fit, alpha):
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import Ridge
    features = pd.DataFrame(dict(dose=np.log1p(meta.dose.to_numpy(float)),
                                 time=meta.time_h.to_numpy(float),
                                 unit=meta.unit.astype(str), level=meta.level.astype(str)))
    encoder = ColumnTransformer([
        ('numeric', StandardScaler(), ['dose', 'time']),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['unit', 'level'])])
    encoder.fit(features.iloc[fit])
    X = encoder.transform(features)
    model = Ridge(alpha=alpha, solver='cholesky').fit(X[fit], Y[fit])
    return model.predict(X), dict(encoder=encoder, model=model)


def time_level_mean(meta, Y, fit, query):
    import numpy as np
    mapping, counts = {}, []
    for key, group in meta.iloc[fit].groupby(['time_h', 'level']):
        per_compound = [Y[g.index.to_numpy()].mean(0) for _, g in group.groupby('compound')]
        mapping[key] = np.mean(per_compound, axis=0)
        counts.append(dict(time_h=float(key[0]), level=str(key[1]), compounds=len(per_compound), targets=len(group)))
    result = []
    for _, row in meta.iloc[query].iterrows():
        key = (row.time_h, row.level)
        if key not in mapping:
            raise ValueError(f'No training time/level group: {key}; no automatic fallback')
        result.append(mapping[key])
    return np.asarray(result), counts


def rmse(Y, pred):
    import numpy as np
    return float(np.sqrt(np.mean((Y.astype(float) - pred) ** 2)))


def prepare_stage(first, data, ids, meta, Y, fit, baseline):
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    rows = np.flatnonzero(np.isin(data.target_ids, ids[fit]))
    pack = first.prepare(data, rows, rows)
    E = np.asarray([pack['source'][data.source_ids[data.target_ids == cid]].numpy().mean(0)
                    for cid in ids], dtype=np.float64)
    C = np.asarray([pack['conditions'][data.target_ids == cid].numpy().mean(0)
                    for cid in ids], dtype=np.float64)
    pca = PCA(n_components=min(PROTOCOL['input_components'], len(fit) - 1, E.shape[1]),
              svd_solver='randomized', random_state=42).fit(E[fit])
    keep = pca.singular_values_ > pca.singular_values_[0] * PROTOCOL['rank_tolerance']
    if not keep.any():
        raise ValueError('No nondegenerate input PCA components')
    scores = pca.transform(E)[:, keep]
    scaler = StandardScaler().fit(scores[fit])
    S = data.structure[[data.compound_to_index[c] for c in meta.compound]]
    X = np.c_[scaler.transform(scores), C, S].astype(np.float32)
    residual = Y[fit] - baseline[fit]
    output = PCA(n_components=min(PROTOCOL['output_components'], len(fit) - 1, Y.shape[1]),
                 svd_solver='randomized', random_state=42).fit(residual)
    okeep = output.singular_values_ > output.singular_values_[0] * PROTOCOL['rank_tolerance']
    if not okeep.any():
        raise ValueError('No nondegenerate output PCA components')
    zfit = output.transform(residual)[:, okeep]
    scale = float(np.sqrt(np.mean(zfit ** 2)))
    if not np.isfinite(scale) or scale <= 1e-8 or not np.isfinite(X).all():
        raise ValueError('Invalid feature or residual scale')
    # No validation/test truth is transformed into training targets.
    Z = torch.zeros(len(ids), int(okeep.sum()))
    Z[fit] = torch.tensor(zfit / scale, dtype=torch.float32)
    state = dict(input_preprocess=pack['preprocess'], input_pca=pca, input_keep=keep,
                 input_scaler=scaler, output_pca=output, output_keep=okeep, output_scale=scale)
    return dict(X=torch.tensor(X), Z=Z, state=state)


def restore(scores, state):
    return (scores * state['output_scale'] @ state['output_pca'].components_[state['output_keep']]
            + state['output_pca'].mean_)


def make_model(input_dim, latent_dim, seed):
    import torch
    from torch import nn
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    class LatentPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = latent_dim
            self.context = nn.Sequential(nn.Linear(input_dim, 64), nn.SiLU())
            self.time = nn.Embedding(PROTOCOL['steps'], 16)
            self.head = nn.Sequential(nn.Linear(64 + latent_dim + 16, 128), nn.SiLU(), nn.Linear(128, latent_dim))
        def forward(self, x, noisy, t):
            return self.head(torch.cat([self.context(x), noisy, self.time(t)], dim=1))
    model = LatentPredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL['learning_rate'],
                                 weight_decay=PROTOCOL['weight_decay'])
    return model, optimizer


def schedule():
    import torch
    T = PROTOCOL['steps']
    u = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    f = torch.cos((u + .008) / 1.008 * torch.pi / 2).square()
    cumulative = f / f[0]
    beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-6, .999).float()
    alpha = 1 - beta
    abar = torch.cumprod(alpha, dim=0)
    return beta, alpha, abar, torch.cat([torch.ones(1), abar[:-1]])


def train_epoch(kind, model, optimizer, pack, fit):
    import torch
    beta, alpha, abar, prev = schedule()
    model.train()
    for local in torch.randperm(len(fit)).split(PROTOCOL['batch_size']):
        rows = fit[local.numpy()]
        truth = pack['Z'][rows]
        t = torch.zeros(len(rows), dtype=torch.long) if kind == 'mlp' else torch.randint(PROTOCOL['steps'], (len(rows),))
        noisy = torch.zeros_like(truth) if kind == 'mlp' else (
            abar[t, None].sqrt() * truth + (1 - abar[t, None]).sqrt() * torch.randn_like(truth))
        loss = (model(pack['X'][rows], noisy, t) - truth).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL['gradient_clip'])
        optimizer.step()


def predict_scores(kind, model, pack, rows, seed):
    import numpy as np
    import torch
    beta, alpha, abar, prev = schedule()
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(rows), PROTOCOL['batch_size']):
            batch = rows[start:start + PROTOCOL['batch_size']]
            context = pack['X'][batch]
            if kind == 'mlp':
                values = model(context, torch.zeros(len(batch), model.latent_dim), torch.zeros(len(batch), dtype=torch.long))
            else:
                context = context.repeat_interleave(PROTOCOL['samples'], dim=0)
                noisy = torch.randn((len(context), model.latent_dim), generator=generator)
                for step in reversed(range(PROTOCOL['steps'])):
                    x0 = model(context, noisy, torch.full((len(context),), step, dtype=torch.long))
                    denominator = 1 - abar[step]
                    noisy = (beta[step] * prev[step].sqrt() / denominator * x0
                             + alpha[step].sqrt() * (1 - prev[step]) / denominator * noisy)
                    if step:
                        variance = beta[step] * (1 - prev[step]) / denominator
                        noisy += variance.sqrt() * torch.randn(noisy.shape, generator=generator)
                values = noisy.reshape(len(batch), PROTOCOL['samples'], model.latent_dim).mean(1)
            array = values.numpy()
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite prediction')
            result.append(array)
    return np.concatenate(result)


def fit_job(job, inner, outer, Y, itr, validation, testing, training, inner_base, final_base, folder):
    import numpy as np
    import pandas as pd
    import torch
    kind, seed = job['model'], job['model_seed']
    model, optimizer = make_model(inner['X'].shape[1], inner['Z'].shape[1], seed)
    best = dict(epoch=0, strength=0., validation_response_rmse=rmse(Y[validation], inner_base[validation]))
    history = [best.copy()]
    for epoch in range(1, PROTOCOL['epochs'] + 1):
        train_epoch(kind, model, optimizer, inner, itr)
        if epoch in PROTOCOL['validation_epochs']:
            correction = restore(predict_scores(kind, model, inner, validation, seed + 20000), inner['state'])
            for strength in PROTOCOL['strengths']:
                record = dict(epoch=epoch, strength=strength,
                              validation_response_rmse=rmse(Y[validation], inner_base[validation] + strength * correction))
                if not np.isfinite(record['validation_response_rmse']):
                    raise ValueError('Nonfinite validation metric')
                history.append(record)
                if record['validation_response_rmse'] < best['validation_response_rmse']:
                    best = record.copy()
            pd.DataFrame(history).to_csv(folder / 'validation_history.csv', index=False)
            print(f'{job} epoch={epoch}: {best}', flush=True)
    save_json(folder / 'selection.json', best)
    prediction = final_base[testing].copy()
    if best['strength'] > 0:
        model, optimizer = make_model(outer['X'].shape[1], outer['Z'].shape[1], seed)
        for epoch in range(1, best['epoch'] + 1):
            train_epoch(kind, model, optimizer, outer, training)
            if epoch == 1 or epoch % 10 == 0 or epoch == best['epoch']:
                print(f'Refit {job}: {epoch}/{best["epoch"]}', flush=True)
        prediction += best['strength'] * restore(
            predict_scores(kind, model, outer, testing, seed + 30000), outer['state'])
        torch.save(model.state_dict(), folder / 'model.pt')
    return prediction, best


def aggregate(out, records, baseline_rows, bootstrap_samples):
    import numpy as np
    import pandas as pd
    all_rows = []
    for record in records:
        detail = pd.read_csv(out / record['metrics_path'])
        for key in ['split_seed', 'fold', 'model_seed', 'model']:
            detail[key] = record['job'][key]
        all_rows.append(detail)
    baseline = pd.DataFrame(baseline_rows)
    for name in ['target_conditions_ridge', 'time_level_mean', 'zero_response']:
        frame = baseline[['split_seed', 'fold', 'condition_id', 'compound', name + '_mse']].copy()
        frame = frame.rename(columns={name + '_mse': 'response_mse'})
        frame['model'] = name
        # One baseline row per split/target: never replicate by model seed.
        frame['model_seed'] = -1
        all_rows.append(frame)
    detail = pd.concat(all_rows, ignore_index=True)
    detail.to_csv(out / 'oof_condition_metrics.csv', index=False)
    units = detail.groupby(['split_seed', 'model_seed', 'model']).response_mse.agg(['sum', 'count']).reset_index()
    units['response_rmse'] = np.sqrt(units['sum'] / units['count'])
    units.to_csv(out / 'pooled_by_split_model_seed.csv', index=False)
    neural = units.loc[units.model.isin(['mlp', 'ddpm'])].pivot(index=['split_seed', 'model_seed'], columns='model', values='response_rmse')
    neural['ddpm_minus_mlp'] = neural.ddpm - neural.mlp
    base = units.loc[units.model_seed.eq(-1)].pivot(index='split_seed', columns='model', values='response_rmse')
    paired = neural.reset_index().merge(base.reset_index(), on='split_seed', validate='many_to_one')
    for model in ['mlp', 'ddpm']:
        for name in ['target_conditions_ridge', 'time_level_mean', 'zero_response']:
            paired[model + '_minus_' + name] = paired[model] - paired[name]
    paired.to_csv(out / 'paired_comparisons.csv', index=False)
    # Average squared errors across repeats/seeds for each target; this is NOT prediction ensembling.
    target = detail.groupby(['condition_id', 'compound', 'model']).response_mse.mean().unstack('model')
    if target.isna().any().any():
        raise ValueError('Incomplete paired target coverage')
    summary = pd.DataFrame(dict(model=target.columns, pooled_rmse=np.sqrt(target.mean()).to_numpy()))
    summary.to_csv(out / 'summary.csv', index=False)
    if bootstrap_samples:
        # Resample compounds, retaining all their targets and averaging seed/repeat MSE first.
        sums = target.groupby(level='compound').sum()
        counts = target.groupby(level='compound').size().reindex(sums.index).to_numpy()
        rng = np.random.default_rng(71442)
        draws = rng.integers(len(sums), size=(bootstrap_samples, len(sums)))
        sampled = np.sqrt(sums.to_numpy()[draws].sum(1) / counts[draws].sum(1)[:, None])
        names = list(sums.columns)
        ci = []
        for comparator in ['mlp', 'target_conditions_ridge', 'time_level_mean']:
            delta = sampled[:, names.index('ddpm')] - sampled[:, names.index(comparator)]
            estimate = float(np.sqrt(target.ddpm.mean()) - np.sqrt(target[comparator].mean()))
            lo, hi = np.quantile(delta, [.025, .975])
            ci.append(dict(comparison='ddpm_minus_' + comparator, estimate=estimate,
                           percentile_95_low=float(lo), percentile_95_high=float(hi),
                           resampling_unit='compound_fixed_predictions_not_refitting'))
        pd.DataFrame(ci).to_csv(out / 'compound_bootstrap.csv', index=False)


def execute(args):
    import numpy as np
    import pandas as pd
    import torch
    import sklearn
    import scipy
    import joblib
    from threadpoolctl import threadpool_limits
    inputs = [args.prepared / n for n in ['vitro_expression.npy', 'vivo_expression.npy', 'vitro_conditions.csv',
              'vivo_conditions.csv', 'pairs.csv.gz', 'probes.csv']] + [args.structure]
    protect_output(args.out, [args.prepared, args.structure])
    contract = dict(protocol=PROTOCOL, arguments={k: str(v.resolve()) if isinstance(v, Path) else v
                    for k, v in vars(args).items() if k != 'execute'},
                    input_sha256={str(p.resolve()): digest(p) for p in inputs},
                    source_sha256={str(p.resolve()): digest(p) for p in [Path(__file__), SNAPSHOT]},
                    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                  torch=torch.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, joblib=joblib.__version__))
    out = args.out.resolve()
    contract_folder(out, contract)
    first, data, ids, meta, Y, taudit, saudit = load_data(args)
    pd.DataFrame(taudit).to_csv(out / 'response_target_audit.csv', index=False)
    pd.DataFrame(saudit).to_csv(out / 'source_delta_audit.csv', index=False)
    data.structure_audit.to_csv(out / 'structure_audit.csv', index=False)
    torch.set_num_threads(args.threads)
    records, baseline_rows = [], []
    with threadpool_limits(limits=args.threads):
        for seed in args.split_seeds:
            mapping = compound_folds(data.names, seed, args.folds)
            pd.DataFrame([dict(compound=n, fold=f) for n, f in mapping.items()]).to_csv(out / f'compound_folds_{seed}.csv', index=False)
            for fold in range(args.folds):
                folder = out / f'split_{seed}' / f'fold_{fold}'
                folder.mkdir(parents=True, exist_ok=True)
                tn, vn = inner_compounds(mapping, fold, seed)
                itr = np.flatnonzero(meta.compound.isin(tn))
                validation = np.flatnonzero(meta.compound.isin(vn))
                testing = np.flatnonzero(meta.compound.map(mapping).eq(fold))
                training = np.sort(np.r_[itr, validation])
                if not all(len(x) for x in [itr, validation, testing]):
                    raise ValueError('Empty split')
                assignments = [dict(compound=n, stage=('outer_test' if f == fold else 'inner_validation' if n in vn else 'inner_train')) for n, f in mapping.items()]
                pd.DataFrame(assignments).to_csv(folder / 'compound_split.csv', index=False)
                bhistory = []
                for alpha in PROTOCOL['baseline_alphas']:
                    pred, _ = baseline_stage(meta, Y, itr, alpha)
                    bhistory.append(dict(alpha=alpha, validation_response_rmse=rmse(Y[validation], pred[validation])))
                chosen = min(bhistory, key=lambda x: x['validation_response_rmse'])['alpha']
                inner_base, inner_ridge = baseline_stage(meta, Y, itr, chosen)
                final_base, final_ridge = baseline_stage(meta, Y, training, chosen)
                mean_pred, counts = time_level_mean(meta, Y, training, testing)
                pd.DataFrame(bhistory).to_csv(folder / 'baseline_validation.csv', index=False)
                save_json(folder / 'time_level_training_counts.json', counts)
                joblib.dump(dict(inner=inner_ridge, final=final_ridge, alpha=chosen), folder / 'ridge.joblib')
                np.savez_compressed(folder / 'baseline_predictions.npz', target_ids=ids[testing], ridge=final_base[testing], time_level=mean_pred)
                for pos, idx in enumerate(testing):
                    baseline_rows.append(dict(split_seed=seed, fold=fold, condition_id=int(ids[idx]), compound=str(meta.iloc[idx].compound),
                        target_conditions_ridge_mse=float(np.mean((Y[idx] - final_base[idx]) ** 2)),
                        time_level_mean_mse=float(np.mean((Y[idx] - mean_pred[pos]) ** 2)),
                        zero_response_mse=float(np.mean(Y[idx] ** 2))))
                pd.DataFrame(baseline_rows).to_csv(out / 'baseline_condition_metrics.csv', index=False)
                inner = prepare_stage(first, data, ids, meta, Y, itr, inner_base)
                outer = prepare_stage(first, data, ids, meta, Y, training, final_base)
                joblib.dump(inner['state'], folder / 'inner_preprocess.joblib')
                joblib.dump(outer['state'], folder / 'final_preprocess.joblib')
                for mseed in args.model_seeds:
                    for kind in ['mlp', 'ddpm']:
                        job = dict(split_seed=seed, fold=fold, model_seed=mseed, model=kind)
                        task = folder / f'{kind}_seed{mseed}'
                        task.mkdir(exist_ok=True)
                        record = read_completion(task, job)
                        if record is None:
                            attempt = task / ('attempt_' + uuid.uuid4().hex)
                            attempt.mkdir()
                            print('Starting', job, flush=True)
                            pred, selected = fit_job(job, inner, outer, Y, itr, validation, testing, training, inner_base, final_base, attempt)
                            np.savez_compressed(attempt / 'predictions.npz', target_ids=ids[testing], prediction=pred)
                            detail = meta.iloc[testing][['condition_id', 'compound', 'time_h', 'level']].copy()
                            detail['response_mse'] = np.mean((Y[testing] - pred) ** 2, axis=1)
                            detail.to_csv(attempt / 'condition_metrics.csv', index=False)
                            metrics = dict(**job, **selected, response_rmse=rmse(Y[testing], pred))
                            save_json(attempt / 'metrics.json', metrics)
                            record = dict(job=job, metrics=metrics,
                                metrics_path=str((attempt / 'condition_metrics.csv').relative_to(out)),
                                sha256={str(p.relative_to(task)): digest(p) for p in attempt.iterdir() if p.is_file()})
                            save_json(task / 'complete.json', record)
                        records.append(record)
                        pd.DataFrame([r['metrics'] for r in records]).to_csv(out / 'results.csv', index=False)
                        print(f'Completed {len(records)}/{len(jobs(args))}: {job}', flush=True)
    if len(records) != len(jobs(args)):
        raise ValueError('Incomplete execution')
    aggregate(out, records, baseline_rows, args.bootstrap_samples)
    save_json(out / 'completed.json', dict(completed_jobs=len(records), expected_jobs=len(jobs(args))))
    print('ALL COMPLETE:', out, flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['prepared', 'structure', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--split-seeds', type=int, nargs='+', default=[42])
    p.add_argument('--model-seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--bootstrap-samples', type=int, default=1000)
    p.add_argument('--execute', action='store_true')
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.folds < 2 or args.threads < 1 or args.bootstrap_samples < 0:
        p.error('Invalid folds, threads or bootstrap count')
    for field in ['split_seeds', 'model_seeds']:
        values = getattr(args, field)
        if len(values) != len(set(values)) or any(n < 0 or n >= 2 ** 32 for n in values):
            p.error('Seeds must be unique integers in [0, 2**32)')
    planned = jobs(args)
    print(json.dumps(dict(mode='execute' if args.execute else 'plan_only',
        split_seeds=args.split_seeds, model_seeds=args.model_seeds, folds=args.folds,
        fold_seed_pairs=len(planned) // 2, neural_jobs=len(planned),
        neural_fits_up_to=len(planned) * 2, protocol=PROTOCOL), indent=2), flush=True)
    if args.execute:
        execute(args)
    return planned


if __name__ == '__main__':
    main()
