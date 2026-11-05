"""Evaluate saved repeated-run predictions. Never trains or loads model/joblib objects."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def within(path, parent):
    path = path.resolve()
    if parent.resolve() not in path.parents:
        raise ValueError(f'Artifact path outside run: {path}')
    return path


def row_correlation(Y, P, ranks=False):
    import numpy as np
    from scipy.stats import rankdata
    if ranks:
        Y, P = rankdata(Y, axis=1), rankdata(P, axis=1)
    a = Y - Y.mean(axis=1, keepdims=True)
    b = P - P.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    result = np.full(len(Y), np.nan)
    np.divide(np.sum(a * b, axis=1), denominator, out=result, where=denominator > 0)
    return np.clip(result, -1., 1.)


def score(Y, P, thresholds):
    import numpy as np
    if Y.shape != P.shape or Y.ndim != 2 or not np.isfinite(Y).all() or not np.isfinite(P).all():
        raise ValueError('Invalid truth/prediction shape or values')
    error = P - Y
    mse = np.mean(error * error, axis=1)
    mae = np.mean(np.abs(error), axis=1)
    pearson = row_correlation(Y, P)
    spearman = row_correlation(Y, P, ranks=True)
    sst = float(np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2))
    regression = dict(n_targets=len(Y), n_probes=Y.shape[1], rmse=float(np.sqrt(mse.mean())),
        mae=float(mae.mean()), r2_probe_variance_weighted=1 - float(np.sum(error * error)) / sst if sst > 0 else None,
        pearson_target_mean=float(np.nanmean(pearson)) if np.isfinite(pearson).any() else None,
        pearson_valid_targets=int(np.isfinite(pearson).sum()),
        spearman_target_mean=float(np.nanmean(spearman)) if np.isfinite(spearman).any() else None,
        spearman_valid_targets=int(np.isfinite(spearman).sum()))
    classification, confusion = [], []
    labels = ['down', 'small', 'up']
    for threshold in thresholds:
        # Classes are amplitude thresholds, NOT statistically significant differential expression.
        truth = np.where(Y >= threshold, 2, np.where(Y <= -threshold, 0, 1))
        pred = np.where(P >= threshold, 2, np.where(P <= -threshold, 0, 1))
        matrix = np.bincount((truth * 3 + pred).ravel(), minlength=9).reshape(3, 3)
        support = matrix.sum(axis=1); predicted = matrix.sum(axis=0); tp = matrix.diagonal()
        precision = np.divide(tp, predicted, out=np.zeros(3, float), where=predicted > 0)
        recall = np.divide(tp, support, out=np.zeros(3, float), where=support > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3, float), where=precision + recall > 0)
        active = np.abs(Y) >= threshold
        active_count = int(active.sum())
        sign_accuracy = float(np.mean(np.sign(P[active]) == np.sign(Y[active]))) if active_count else None
        record = dict(threshold=threshold, accuracy=float(tp.sum() / matrix.sum()), macro_f1=float(f1.mean()),
            weighted_f1=float(np.sum(f1 * support) / support.sum()),
            active_sign_accuracy=sign_accuracy, n_active=active_count,
            actual_active_fraction=float(active.mean()),
            predicted_active_fraction=float(np.mean(np.abs(P) >= threshold)))
        for i, name in enumerate(labels):
            record.update({f'{name}_precision': float(precision[i]), f'{name}_recall': float(recall[i]),
                           f'{name}_f1': float(f1[i]), f'{name}_support': int(support[i])})
            for j, pname in enumerate(labels):
                confusion.append(dict(threshold=threshold, actual=name, predicted=pname, count=int(matrix[i, j])))
        classification.append(record)
    return regression, classification, confusion, dict(rmse=np.sqrt(mse), mae=mae, pearson=pearson, spearman=spearman)


def truth_for_run(run):
    import numpy as np
    import pandas as pd
    info = json.loads((run / 'run.json').read_text())
    done = json.loads((run / 'completed.json').read_text())
    families = info['arguments'].get('families', ['mlp', 'ddpm'])
    expected = len(info['arguments']['split_seeds']) * info['arguments']['folds'] * len(info['arguments']['model_seeds']) * len(families)
    if done.get('completed_jobs') != expected or done.get('expected_jobs') != expected:
        raise ValueError('Repeated run is incomplete or has an unsupported completion record')
    # Verify all original input hashes before reading truth. Do not load source expressions/models.
    for filename, expected_hash in info['input_sha256'].items():
        if digest(Path(filename)) != expected_hash:
            raise ValueError(f'Input checksum changed: {filename}')
    prepared = Path(info['arguments']['prepared'])
    vo = pd.read_csv(prepared / 'vivo_conditions.csv')
    raw = np.load(prepared / 'vivo_expression.npy', mmap_mode='r', allow_pickle=False)
    if not np.array_equal(vo.condition_id, np.arange(len(raw))):
        raise ValueError('Target row alignment mismatch')
    pairs = pd.read_csv(prepared / 'pairs.csv.gz', usecols=['condition_id_target', 'level_target'])
    ids = np.sort(pairs.loc[~pairs.level_target.eq('Control'), 'condition_id_target'].unique().astype(int))
    if len(ids) == 0 or ids.min() < 0 or ids.max() >= len(raw):
        raise ValueError('Invalid target IDs')
    controls = vo.loc[vo.level.eq('Control')]
    if controls.duplicated(['compound', 'unit', 'time_h']).any():
        raise ValueError('Ambiguous controls')
    lookup = controls.set_index(['compound', 'unit', 'time_h']).condition_id.to_dict()
    control_ids = []
    for cid in ids:
        row = vo.iloc[cid]
        key = (row.compound, row.unit, row.time_h)
        if key not in lookup:
            raise ValueError(f'Matched target control missing: {key}')
        control_ids.append(int(lookup[key]))
    # Keep original subtraction dtype to match the trainer's response definition.
    Y = np.asarray(raw[ids] - raw[np.asarray(control_ids)], dtype=np.float64)
    meta = vo.iloc[ids].reset_index(drop=True)
    return info, ids, meta, Y


def load_model_prediction(run, seed, fold, model_seed, kind, audit):
    import numpy as np
    task = run / f'split_{seed}' / f'fold_{fold}' / f'{kind}_seed{model_seed}'
    marker = task / 'complete.json'
    record = json.loads(marker.read_text())
    expected = dict(split_seed=seed, fold=fold, model_seed=model_seed, model=kind)
    if record['job'] != expected:
        raise ValueError('Job marker mismatch')
    paths = []
    for relative, expected_hash in record['sha256'].items():
        p = within(task / relative, task)
        if digest(p) != expected_hash:
            raise ValueError(f'Completed artifact changed: {p}')
        audit[str(p)] = expected_hash
        if p.name == 'predictions.npz':
            paths.append(p)
    if len(paths) != 1:
        raise ValueError('Expected exactly one completed prediction file')
    audit[str(marker)] = digest(marker)
    with np.load(paths[0], allow_pickle=False) as saved:
        return saved['target_ids'].copy(), saved['prediction'].astype(np.float64)


def execute(args):
    import numpy as np
    import pandas as pd
    import scipy
    run, out = args.run.resolve(), args.out.resolve()
    if out == run or out in run.parents or run in out.parents:
        raise ValueError('Evaluation output must be separate from training run')
    if out.exists():
        raise ValueError('Use a new evaluation output directory')
    info, ids, meta, Y = truth_for_run(run)
    prepared = Path(info['arguments']['prepared']).resolve()
    if out == prepared or out in prepared.parents or prepared in out.parents:
        raise ValueError('Evaluation output must be separate from source data')
    out.mkdir(parents=True)
    args_run = info['arguments']
    audit = {}
    regression, classification, confusions, target_rows = [], [], [], []
    direct = args_run.get('prediction_target', 'ridge_residual') == 'direct_response'
    baselines = ['time_level_mean', 'zero_response'] if direct else ['target_conditions_ridge', 'time_level_mean', 'zero_response']
    units = [(seed, -1, kind) for seed in args_run['split_seeds'] for kind in baselines]
    units += [(seed, m, kind) for seed in args_run['split_seeds'] for m in args_run['model_seeds'] for kind in args_run.get('families', ['mlp', 'ddpm'])]
    id_position = {int(cid): i for i, cid in enumerate(ids)}
    for number, (seed, model_seed, kind) in enumerate(units, 1):
        prediction = np.empty_like(Y)
        seen = np.zeros(len(ids), bool)
        fold_file = run / f'compound_folds_{seed}.csv'
        assignments = pd.read_csv(fold_file)
        if assignments.compound.duplicated().any() or set(assignments.compound) != set(meta.compound):
            raise ValueError('Compound fold coverage differs')
        mapping = assignments.set_index('compound').fold.to_dict()
        audit[str(fold_file)] = digest(fold_file)
        for fold in range(args_run['folds']):
            expected_ids = set(ids[meta.compound.map(mapping).eq(fold).to_numpy()])
            if kind in baselines:
                file = run / f'split_{seed}' / f'fold_{fold}' / 'baseline_predictions.npz'
                audit[str(file)] = digest(file)
                with np.load(file, allow_pickle=False) as saved:
                    target_ids = saved['target_ids'].copy()
                    p = (np.zeros((len(target_ids), Y.shape[1])) if kind == 'zero_response'
                         else saved['ridge' if kind == 'target_conditions_ridge' else 'time_level'].astype(float))
            else:
                target_ids, p = load_model_prediction(run, seed, fold, model_seed, kind, audit)
            if target_ids.dtype.kind not in 'iu' or len(set(target_ids)) != len(target_ids) or set(target_ids) != expected_ids:
                raise ValueError('Fold target coverage or ID dtype mismatch')
            if p.shape != (len(target_ids), Y.shape[1]) or not np.isfinite(p).all():
                raise ValueError('Invalid prediction array')
            positions = np.array([id_position[int(cid)] for cid in target_ids])
            if seen[positions].any():
                raise ValueError('Duplicate OOF targets')
            seen[positions] = True
            prediction[positions] = p
            # Recompute original MSE to ensure truth definition agrees with the saved report.
            if kind not in baselines:
                task = run / f'split_{seed}' / f'fold_{fold}' / f'{kind}_seed{model_seed}'
                marker = json.loads((task / 'complete.json').read_text())
                metrics_path = within(run / marker['metrics_path'], task)
                if str(metrics_path) not in audit:
                    raise ValueError('Condition metrics not covered by completion checksum')
                saved_metrics = pd.read_csv(metrics_path).set_index('condition_id')
                if set(saved_metrics.index) != set(target_ids) or not saved_metrics.index.is_unique:
                    raise ValueError('Condition metric target coverage mismatch')
                mse = np.mean((Y[positions] - p) ** 2, axis=1)
                np.testing.assert_allclose(mse, saved_metrics.loc[target_ids, 'response_mse'], rtol=1e-6, atol=1e-10)
        if not seen.all():
            raise ValueError('OOF predictions do not cover all targets')
        reg, cls, cm, detail = score(Y, prediction, args.thresholds)
        unit = dict(split_seed=seed, model_seed=model_seed, model=kind)
        regression.append(dict(**unit, **reg))
        classification.extend(dict(**unit, **x) for x in cls)
        confusions.extend(dict(**unit, **x) for x in cm)
        target = meta[['condition_id', 'compound', 'time_h', 'level']].copy()
        for k, v in detail.items():
            target[k] = v
        for k, v in unit.items():
            target[k] = v
        target_rows.append(target)
        print(f'Evaluated {number}/{len(units)}: {unit}; RMSE={reg["rmse"]:.6f}', flush=True)
    reg = pd.DataFrame(regression); cls = pd.DataFrame(classification)
    prediction_target = args_run.get('prediction_target', 'ridge_residual')
    reg.insert(0, 'prediction_target', prediction_target)
    cls.insert(0, 'prediction_target', prediction_target)
    for metric in ['accuracy', 'macro_f1', 'weighted_f1', 'active_sign_accuracy', 'actual_active_fraction', 'predicted_active_fraction',
                   'down_precision', 'down_recall', 'down_f1', 'small_precision', 'small_recall', 'small_f1', 'up_precision', 'up_recall', 'up_f1']:
        cls[metric + '_pct'] = cls[metric] * 100
    for seed in args_run['split_seeds']:
        baseline_rmse = float(reg.loc[(reg.split_seed == seed) & reg.model.eq('time_level_mean'), 'rmse'].iloc[0])
        ridge_values = reg.loc[(reg.split_seed == seed) & reg.model.eq('target_conditions_ridge'), 'rmse']
        ridge_rmse = float(ridge_values.iloc[0]) if len(ridge_values) else np.nan
        rows = reg.split_seed.eq(seed)
        reg.loc[rows, 'rmse_improvement_vs_time_level_pct'] = 100 * (1 - reg.loc[rows, 'rmse'] / baseline_rmse) if baseline_rmse > 0 else np.nan
        reg.loc[rows, 'rmse_improvement_vs_ridge_pct'] = 100 * (1 - reg.loc[rows, 'rmse'] / ridge_rmse) if ridge_rmse > 0 else np.nan
    reg.to_csv(out / 'regression_by_unit.csv', index=False)
    cls.to_csv(out / 'classification_by_unit.csv', index=False)
    pd.DataFrame(confusions).to_csv(out / 'confusion_matrices.csv', index=False)
    pd.concat(target_rows, ignore_index=True).to_csv(out / 'target_metrics.csv', index=False)
    reg_metrics = ['rmse', 'mae', 'r2_probe_variance_weighted', 'pearson_target_mean', 'spearman_target_mean',
                   'rmse_improvement_vs_ridge_pct', 'rmse_improvement_vs_time_level_pct']
    cls_metrics = ['accuracy_pct', 'macro_f1_pct', 'active_sign_accuracy_pct', 'down_f1_pct', 'small_f1_pct', 'up_f1_pct',
                   'actual_active_fraction_pct', 'predicted_active_fraction_pct']
    reg_summary = reg.groupby('model')[reg_metrics].agg(['mean', 'std'])
    cls_summary = cls.groupby(['threshold', 'model'])[cls_metrics].agg(['mean', 'std'])
    for table, name in [(reg_summary, 'regression_summary.csv'), (cls_summary, 'classification_summary.csv')]:
        table.columns = ['_'.join(c) for c in table.columns]
        table.insert(0, 'prediction_target', prediction_target)
        table.to_csv(out / name)
    manifest = dict(training_run=str(run), prediction_target=prediction_target, thresholds=args.thresholds, training_run_sha256=digest(run / 'run.json'),
        evaluator_sha256=digest(Path(__file__)), original_inputs_verified=info['input_sha256'],
        artifacts_verified=audit, training_performed=False, checkpoint_or_joblib_loaded=False,
        threshold_interpretation='exploratory_amplitude_not_statistical_DE_or_verified_log2_fold_change',
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))
    (out / 'evaluation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'completed.json').write_text(json.dumps(dict(evaluated_units=len(units))) + '\n')
    print('EVALUATION COMPLETE:', out, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--thresholds', type=float, nargs='+', default=[.5, 1.])
    args = p.parse_args(argv)
    import math
    if len(set(args.thresholds)) != len(args.thresholds) or any(not math.isfinite(t) or t <= 0 for t in args.thresholds):
        p.error('Thresholds must be unique, positive and finite')
    execute(args)


if __name__ == '__main__':
    main()
