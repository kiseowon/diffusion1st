"""Plan or run repeated structure-on sparse IVIVE comparisons in a new directory.

Default is plan-only: no inputs read, output created, mask generated, or training.
Original single-run code and results remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

FAMILIES = ('MLP', 'cGAN', 'epsilon-DDPM')


def jobs(split_seeds, model_seeds, graph_seeds, families=FAMILIES):
    result = []
    for split in split_seeds:
        for model in model_seeds:
            for family in families:
                for mode, graph in [('none', None), ('sparse', None)] + [
                        ('random_sparse', g) for g in graph_seeds]:
                    result.append(dict(split_seed=split, model_seed=model, family=family,
                                       pathway_mode=mode, graph_seed=graph))
    return result


def compound_split(names, seed):
    import numpy as np
    import pandas as pd
    names = np.random.default_rng(seed).permutation(sorted(names))
    nt = max(1, int(np.ceil(.2 * len(names))))
    rest = names[nt:]
    nv = max(1, int(np.ceil(.2 * len(rest))))
    groups = [('inner_train', rest[nv:]), ('inner_validation', rest[:nv]),
              ('outer_test', names[:nt])]
    if len(rest[nv:]) < 2:
        raise ValueError('Too few compounds')
    return pd.DataFrame([dict(compound=n, stage=stage)
                         for stage, group in groups for n in sorted(group)])


def summarize(table):
    """Average masks within split/model first; shared baselines are never duplicated."""
    import pandas as pd
    metrics = ['absolute_rmse', 'response_rmse']
    unit = ['split_seed', 'model_seed', 'family']
    per_unit = table.groupby(unit + ['pathway_mode'])[metrics].mean().reset_index()
    paired = []
    for key, group in per_unit.groupby(unit):
        indexed = group.set_index('pathway_mode')
        if set(indexed.index) != {'none', 'sparse', 'random_sparse'}:
            raise ValueError('Incomplete comparisons; no final summary written')
        for reference in ['none', 'random_sparse']:
            record = dict(zip(unit, key), reference=reference)
            for metric in metrics:
                record[metric + '_delta'] = float(indexed.loc['sparse', metric]
                                                 - indexed.loc[reference, metric])
            paired.append(record)
    paired = pd.DataFrame(paired)
    # Model repeats are also nested inside a split, not extra independent test sets.
    split_paired = paired.groupby(['split_seed', 'family', 'reference'])[
        [m + '_delta' for m in metrics]].mean().reset_index()
    summary = split_paired.groupby(['family', 'reference'])[
        [m + '_delta' for m in metrics]].agg(['count', 'mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    return per_unit, paired, split_paired, summary.reset_index()


def training_mean_response(data, train_rows, test_rows):
    """Level-specific mean: unique targets -> per compound -> across compounds.

    Training is outer-refit compounds (inner train + validation); no fitted
    parameter uses test expression. Evaluation weights conditions/probes equally,
    as the existing response RMSE does. No absolute-expression metric is defined.
    """
    import numpy as np
    import pandas as pd
    table = data.vo
    train_ids = np.unique(data.target_ids[train_rows])
    test_ids = np.unique(data.target_ids[test_rows])
    if set(table.iloc[train_ids].compound) & set(table.iloc[test_ids].compound):
        raise ValueError('Baseline train/test compound overlap')
    controls = table.loc[table.level.eq('Control')]
    keys = ['compound', 'unit', 'time_h']
    if controls.duplicated(keys).any():
        raise ValueError('Ambiguous baseline control')
    lookup = controls.set_index(keys).condition_id.to_dict()

    def responses(ids):
        available = set(ids)
        records, vectors = [], []
        for cid in ids:
            row = table.iloc[cid]
            if row.level == 'Control':
                continue
            if pd.isna(row.level) or not str(row.level).strip():
                raise ValueError('Missing target dose level')
            control = lookup.get((row.compound, row.unit, row.time_h))
            if control is None or control not in available:
                raise ValueError('Matched control absent from selected targets')
            truth = data.target[cid].astype(float) - data.target[control].astype(float)
            if not np.isfinite(truth).all():
                raise ValueError('Nonfinite response')
            vectors.append(truth)
            records.append(dict(compound=row.compound, level=str(row.level),
                                condition_id=int(cid), control_id=int(control)))
        if not vectors:
            raise ValueError('No treated conditions for response baseline')
        return np.stack(vectors), pd.DataFrame(records)

    training, training_meta = responses(train_ids)
    means, counts = {}, {}
    for level, group in training_meta.groupby('level', sort=True):
        compound_means = [training[part.index.to_numpy()].mean(axis=0)
                          for _, part in group.groupby('compound', sort=True)]
        means[level] = np.stack(compound_means).mean(axis=0)
        counts[level] = dict(compounds=len(compound_means), unique_treated_targets=len(group))
    testing, detail = responses(test_ids)
    missing = set(detail.level) - set(means)
    if missing:
        raise ValueError('No training baseline for target dose level: ' + ', '.join(sorted(missing)))
    predicted = np.stack([means[level] for level in detail.level])
    detail['training_mean_response_mse'] = ((testing - predicted) ** 2).mean(axis=1)
    detail['zero_response_mse'] = (testing ** 2).mean(axis=1)
    return means, dict(training_mean_response_rmse=float(np.sqrt(detail.training_mean_response_mse.mean())),
                       zero_response_rmse=float(np.sqrt(detail.zero_response_mse.mean())),
                       n_training_treated=len(training), n_targets=len(test_ids), n_treated=len(testing),
                       baseline_definition='target_level: mean within compound then mean across compounds',
                       training_counts_by_level=counts), detail


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def fit_job(job, data, inner, outer, rows, membership, a, folder):
    import numpy as np
    import pandas as pd
    import torch
    import first_structure_mlp as first
    import sparse_pathway_information_matrix as generative
    from pathway_pattern_mlp import PatternMLP
    itr, iv, tr, te = rows
    mode = 'none' if job['pathway_mode'] == 'none' else 'sparse'
    kind = {'cGAN': 'cgan', 'epsilon-DDPM': 'ddpm'}.get(job['family'])
    settings = SimpleNamespace(seed=job['model_seed'], ddpm_steps=a.ddpm_steps,
                               samples=a.samples, batch_size=a.batch_size)

    def make(pack):
        if kind:
            generative.MEMBERSHIP = membership
            return generative.make_models(kind, data, pack, mode, settings)
        first.seed_all(job['model_seed'])
        model = PatternMLP(pack['gene_dim'], pack['conditions'].shape[1],
                           pack['structure'].shape[1], data.target.shape[1], membership, mode)
        return model, torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01), None, None

    def epoch(model, opt, disc, dopt, pack, training):
        if kind:
            return generative.train_epoch(kind, model, opt, disc, dopt, data,
                                          pack, training, True, settings)
        return first.train_epoch(model, opt, data, pack, training, True, a.batch_size)

    def predict(model, pack, selected, offset):
        if kind:
            return generative.predict(kind, model, data, pack, selected, True,
                                       settings, job['model_seed'] + offset)
        return first.predict(model, data, pack, selected, True, a.batch_size)

    model, opt, disc, dopt = make(inner)
    best, selected_epoch, history = float('inf'), None, []
    for number in range(1, a.epochs + 1):
        loss = epoch(model, opt, disc, dopt, inner, itr)
        if number == 1 or number % a.check_every == 0 or number == a.epochs:
            ids, prediction, _ = predict(model, inner, iv, 20000)
            mse = float(np.mean((prediction.astype(float) - data.target[ids]) ** 2))
            if not np.isfinite(mse):
                raise ValueError('Nonfinite validation score')
            if mse < best:
                best, selected_epoch = mse, number
                torch.save(model.state_dict(), folder / 'inner_best.pt')
            history.append(dict(epoch=number, loss=loss, validation_rmse=np.sqrt(mse)))
            pd.DataFrame(history).to_csv(folder / 'validation_history.csv', index=False)
    # Fresh initialization and preprocessing fitted to train+validation only.
    model, opt, disc, dopt = make(outer)
    for number in range(selected_epoch):
        epoch(model, opt, disc, dopt, outer, tr)
    ids, prediction, pair_rmse = predict(model, outer, te, 30000)
    metrics, detail = first.evaluate(data, ids, prediction)
    np.savez_compressed(folder / 'predictions.npz', target_ids=ids, prediction=prediction)
    detail.to_csv(folder / 'response_condition_metrics.csv', index=False)
    torch.save(model.state_dict(), folder / 'model.pt')
    if disc is not None:
        torch.save(disc.state_dict(), folder / 'discriminator.pt')
    metrics.update(job, selected_epoch=selected_epoch, validation_rmse=float(np.sqrt(best)),
                   pair_rmse=pair_rmse, structure=True, attempt=str(folder))
    # Written last; an interrupted attempt is never treated as complete.
    save_json(folder / 'metrics.json', metrics)
    return metrics


def execute(a, planned):
    import numpy as np
    import pandas as pd
    import torch
    import joblib
    import sklearn
    import platform
    from threadpoolctl import threadpool_limits
    import first_structure_mlp as first
    import pathway_structure_mlp as pathway
    from pathway_random_control import shuffle_membership
    # Protect original inputs and outputs against overlapping output paths.
    out = a.out.resolve()
    for source in [a.prepared, a.annotation, a.structure]:
        source = source.resolve()
        if out == source or out in source.parents or source in out.parents:
            raise ValueError('Output must be separate from input locations')
    inputs = [a.prepared / n for n in ['vitro_expression.npy', 'vivo_expression.npy',
              'vitro_conditions.csv', 'vivo_conditions.csv', 'pairs.csv.gz', 'probes.csv']]
    inputs += [a.structure] + [a.annotation / n for n in
                              ['projection.npy', 'pathway_audit.csv', 'provenance.json']]
    sources = [Path(__file__)] + [Path(__file__).with_name(n) for n in
               ['first_structure_mlp.py', 'pathway_structure_mlp.py', 'pathway_pattern_mlp.py',
                'pathway_random_control.py', 'sparse_pathway_information_matrix.py']]
    config = {k: str(v.resolve()) if isinstance(v, Path) else v
              for k, v in vars(a).items() if k != 'execute'}
    config['hashes'] = {str(p.resolve()): digest(p) for p in inputs + sources}
    config['versions'] = dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                              sklearn=sklearn.__version__, joblib=joblib.__version__,
                              python=platform.python_version())
    if out.exists() and any(out.iterdir()):
        if not (out / 'run.json').is_file() or json.loads((out / 'run.json').read_text()) != config:
            raise ValueError('Existing output has a different configuration, inputs, source, or environment')
    else:
        out.mkdir(parents=True, exist_ok=True)
        save_json(out / 'run.json', config)
    data = first.Data(a)
    projection, pathways, provenance = pathway.load_projection(a.annotation, a.prepared)
    masks = {}
    for seed in ([] if a.baseline_only else a.graph_seeds):
        file = out / f'mask_{seed}.npy'
        if file.exists():
            mask = np.load(file, allow_pickle=False)
            audit = json.loads((out / f'mask_{seed}_audit.json').read_text())
            if digest(file) != audit['mask_file_sha256']:
                raise ValueError('Saved mask hash mismatch')
        else:
            mask, audit = shuffle_membership(projection, seed, a.swaps_per_edge)
            mask = mask.astype(np.uint8)
            np.save(file, mask)
            audit['mask_file_sha256'] = digest(file)
            save_json(out / f'mask_{seed}_audit.json', audit)
        np.testing.assert_array_equal(mask.sum(0), (projection > 0).sum(0))
        np.testing.assert_array_equal(mask.sum(1), (projection > 0).sum(1))
        if mask.shape != projection.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError('Invalid saved mask')
        masks[seed] = mask
    records, baseline_records = [], []
    torch.set_num_threads(a.threads)
    with threadpool_limits(limits=a.threads):
        for seed in a.split_seeds:
            folder = out / f'split_{seed}'
            folder.mkdir(exist_ok=True)
            split = compound_split(data.names, seed)
            split.to_csv(folder / 'compound_split.csv', index=False)
            def selected(stage):
                return np.flatnonzero(data.pairs.compound.isin(split.loc[split.stage.eq(stage), 'compound']))
            itr, iv, te = selected('inner_train'), selected('inner_validation'), selected('outer_test')
            tr = np.sort(np.r_[itr, iv])
            mean, baseline, detail = training_mean_response(data, tr, te)
            baseline['split_seed'] = seed
            levels = sorted(mean)
            np.savez_compressed(folder / 'training_mean_response_by_level.npz',
                                levels=np.asarray(levels), response=np.stack([mean[level] for level in levels]))
            detail.to_csv(folder / 'baseline_response_conditions.csv', index=False)
            save_json(folder / 'baseline_metrics.json', baseline)
            baseline_records.append(baseline)
            pd.DataFrame(baseline_records).to_csv(out / 'response_baselines.csv', index=False)
            if a.baseline_only:
                continue
            inner = pathway.prepare_pathways(data, itr, iv, projection, pathways, provenance)
            outer = pathway.prepare_pathways(data, tr, te, projection, pathways, provenance)
            joblib.dump(outer['preprocess'], folder / 'preprocess.joblib')
            for job in [j for j in planned if j['split_seed'] == seed]:
                key = f"{job['family']}_{job['pathway_mode']}_model{job['model_seed']}_graph{job['graph_seed']}"
                task = folder / key
                task.mkdir(exist_ok=True)
                marker = task / 'complete.json'
                if marker.exists():
                    record = json.loads(marker.read_text())
                    for field, value in job.items():
                        if record.get(field) != value:
                            raise ValueError('Completed job identity mismatch')
                    attempt = Path(record['attempt'])
                    if not (attempt / 'predictions.npz').is_file() or not (attempt / 'metrics.json').is_file():
                        raise ValueError('Completed artifacts missing')
                else:
                    attempt = task / ('attempt_' + uuid.uuid4().hex)
                    attempt.mkdir()
                    print('Starting', job, flush=True)
                    membership = masks[job['graph_seed']] if job['pathway_mode'] == 'random_sparse' else projection
                    record = fit_job(job, data, inner, outer, (itr, iv, tr, te), membership, a, attempt)
                    save_json(marker, record)
                records.append(record)
                pd.DataFrame(records).to_csv(out / 'results.csv', index=False)
    table = pd.DataFrame(records)
    if len(table) != len(planned):
        raise ValueError('Incomplete run')
    if a.baseline_only:
        save_json(out / 'completed.json', dict(mode='baseline_only', splits=len(baseline_records), model_fits=0))
        return
    for name, frame in zip(['per_unit', 'paired_differences', 'split_differences', 'descriptive_summary'], summarize(table)):
        frame.to_csv(out / f'{name}.csv', index=False)
    per_unit = summarize(table)[0]
    against = per_unit.merge(pd.DataFrame(baseline_records), on='split_seed', validate='many_to_one')
    against['response_delta_vs_training_mean'] = against.response_rmse - against.training_mean_response_rmse
    against['response_delta_vs_zero'] = against.response_rmse - against.zero_response_rmse
    against.to_csv(out / 'models_vs_response_baselines.csv', index=False)
    save_json(out / 'completed.json', dict(expected_jobs=len(planned), completed_jobs=len(table)))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['prepared', 'structure', 'annotation', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--split-seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--model-seeds', type=int, nargs='+', default=[42])
    p.add_argument('--graph-seeds', type=int, nargs='+', default=[20261102, 20261103, 20261104])
    for name, default in [('epochs', 30), ('check-every', 5), ('batch-size', 128),
                          ('threads', 4), ('samples', 4), ('ddpm-steps', 100), ('swaps-per-edge', 10)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--families', nargs='+', choices=['mlp', 'cgan', 'ddpm'], default=['mlp'])
    p.add_argument('--baseline-only', action='store_true', help='Only training-mean and zero response baselines; no model fits or masks')
    p.add_argument('--execute', action='store_true', help='Explicitly start new training; omitted = plan only')
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    for field in ['split_seeds', 'model_seeds', 'graph_seeds']:
        values = getattr(a, field)
        if len(values) != len(set(values)) or any(v < 0 or v >= 2**32 for v in values):
            p.error('Seeds must be distinct integers in [0, 2**32)')
    if min(a.epochs, a.check_every, a.batch_size, a.threads, a.samples, a.ddpm_steps, a.swaps_per_edge) < 1:
        p.error('Settings must be positive')
    if len(a.families) != len(set(a.families)):
        p.error('Families must be distinct')
    planned = [] if a.baseline_only else jobs(a.split_seeds, a.model_seeds, a.graph_seeds,
                [{'mlp': 'MLP', 'cgan': 'cGAN', 'ddpm': 'epsilon-DDPM'}[f] for f in a.families])
    print(json.dumps(dict(mode='execute' if a.execute else 'plan_only', split_seeds=a.split_seeds,
                         model_seeds=a.model_seeds, graph_seeds=a.graph_seeds,
                         families=a.families, baseline_only=a.baseline_only,
                         baseline_splits=len(a.split_seeds), jobs=len(planned), fits_up_to=2 * len(planned)), indent=2))
    if a.execute:
        execute(a, planned)
    return planned


if __name__ == '__main__':
    main()
