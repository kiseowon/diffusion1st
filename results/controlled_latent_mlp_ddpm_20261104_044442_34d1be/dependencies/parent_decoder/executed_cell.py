# 저차원 실험을 완료한 같은 커널에서 실행
# 실제 학습 시작: 3 decoder × 3 model seeds = 9조건
# CPU 사용. 기존 결과 덮어쓰기 없음.

required = [
    "data", "Y", "ids", "meta", "IT", "IV", "TR", "TE",
    "projection", "PREPARED", "OUT", "first", "SCRIPTS",
    "save_json", "digest",
]
missing = [k for k in required if k not in globals()]
if missing:
    raise RuntimeError(
        "앞의 저차원 실험 커널이 필요합니다: " + ", ".join(missing)
    )

from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import copy
import json
import uuid
import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from pathway_random_control import shuffle_membership

PARENT = Path(OUT)
RUN = PARENT.parent / (
    "pathway_decoder_pilot_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_" + uuid.uuid4().hex[:6]
)
RUN.mkdir(exist_ok=False)

SETTINGS = dict(
    model_seeds=[42, 43, 44],
    graph_seed=20261102,
    baseline_alphas=[10.0, 100.0, 1000.0],
    input_pca_max=64,
    input_pca_relative_tolerance=1e-8,
    hidden=128,
    free_route_rank=8,
    learning_rate=0.0003,
    weight_decay=0.01,
    dropout=0.1,
    batch_size=64,
    epochs=30,
    validation_epochs=[1, 5, 10, 15, 20, 25, 30],
    strengths=[0.0, 0.25, 0.5, 1.0],
    threads=4,
    baseline="target_conditions_ridge",
    target="direct_vivo_treated_minus_control",
    source="vitro_treated_minus_control",
    residual="in_sample_training_baseline_residual",
    exploratory=True,
)
save_json(RUN / "config.json", SETTINGS)
(RUN / "executed_cell.py").write_text(
    get_ipython().history_manager.input_hist_raw[-1],
    encoding="utf-8",
)
for name in ["run.json", "compound_split.csv", "response_target_audit.csv"]:
    if (PARENT / name).is_file():
        shutil.copy2(PARENT / name, RUN / ("parent_" + name))

shutil.copy2(
    SCRIPTS / "pathway_random_control.py",
    RUN / "pathway_random_control.py",
)

# -------------------------------
# 1. KEGG / degree-preserving random
# -------------------------------
membership = (np.asarray(projection) > 0).astype(np.uint8)
if membership.shape[0] != Y.shape[1]:
    raise ValueError("KEGG probe 차원 불일치")

random_membership, random_audit = shuffle_membership(
    projection, SETTINGS["graph_seed"]
)
random_membership = np.asarray(random_membership).astype(np.uint8)

if not np.isin(random_membership, [0, 1]).all():
    raise ValueError("무작위 mask는 이진 행렬이어야 합니다.")
np.testing.assert_array_equal(
    random_membership.sum(0), membership.sum(0)
)
np.testing.assert_array_equal(
    random_membership.sum(1), membership.sum(1)
)
np.save(RUN / "kegg_membership.npy", membership)
np.save(RUN / "random_membership.npy", random_membership)
save_json(RUN / "randomization_audit.json", random_audit)

print("KEGG shape:", membership.shape)
print("연결 수:", int(membership.sum()))
print("KEGG 미매핑 probe 수:", int((membership.sum(1) == 0).sum()))

# -------------------------------
# 2. source도 treated-control로 변환
# -------------------------------
vitro = pd.read_csv(PREPARED / "vitro_conditions.csv")
raw_source = np.load(PREPARED / "vitro_expression.npy")
np.testing.assert_array_equal(
    vitro.condition_id.to_numpy(), np.arange(len(raw_source))
)
if raw_source.shape != data.source.shape:
    raise ValueError("source shape 불일치")

source_controls = vitro.loc[vitro.level.eq("Control")]
keys = ["compound", "unit", "time_h"]
if source_controls.duplicated(keys).any():
    raise ValueError("in vitro 대조군 연결이 모호합니다.")
source_control_map = (
    source_controls.set_index(keys).condition_id.to_dict()
)

delta_source = np.zeros_like(raw_source, dtype=np.float32)
source_audit = []
for sid in np.unique(data.source_ids):
    row = vitro.iloc[sid]
    key = (row.compound, row.unit, row.time_h)
    if key not in source_control_map:
        raise ValueError(f"in vitro 대조군 없음: {key}")
    control_id = int(source_control_map[key])
    delta_source[sid] = raw_source[sid] - raw_source[control_id]
    source_audit.append(dict(
        condition_id=int(sid),
        control_condition_id=control_id,
        compound=row.compound,
    ))
pd.DataFrame(source_audit).to_csv(
    RUN / "source_delta_audit.csv", index=False
)

delta_data = copy.copy(data)
delta_data.source = delta_source

# 화합물 분할 중첩 검사
sets = [set(meta.iloc[p].compound) for p in [IT, IV, TE]]
if any(sets[a] & sets[b] for a in range(3) for b in range(a)):
    raise ValueError("학습·검증·시험 화합물 중첩")

# -------------------------------
# 3. 목표 조건만 사용하는 Ridge 기준선
# -------------------------------
target_features = pd.DataFrame({
    "dose": np.log1p(meta.dose.to_numpy(float)),
    "time": meta.time_h.to_numpy(float),
    "unit": meta.unit.astype(str).to_numpy(),
    "level": meta.level.astype(str).to_numpy(),
})

def fit_baseline(fit, alpha):
    encoder = ColumnTransformer([
        ("numeric", StandardScaler(), ["dose", "time"]),
        ("category",
         OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["unit", "level"]),
    ])
    encoder.fit(target_features.iloc[fit])
    X = encoder.transform(target_features)
    model = Ridge(alpha=alpha, solver="cholesky")
    model.fit(X[fit], Y[fit])
    prediction = model.predict(X)
    if not np.isfinite(prediction).all():
        raise ValueError("Ridge 예측 NaN/Inf")
    return prediction, dict(encoder=encoder, model=model)

def rmse(actual, prediction):
    return float(np.sqrt(np.mean((actual - prediction) ** 2)))

torch.set_num_threads(SETTINGS["threads"])

with threadpool_limits(limits=SETTINGS["threads"]):
    baseline_history = []
    for alpha in SETTINGS["baseline_alphas"]:
        pred, _ = fit_baseline(IT, alpha)
        baseline_history.append(dict(
            alpha=alpha,
            validation_response_rmse=rmse(Y[IV], pred[IV]),
        ))
    best_alpha = min(
        baseline_history,
        key=lambda r: r["validation_response_rmse"]
    )["alpha"]

    inner_base, _ = fit_baseline(IT, best_alpha)
    final_base, baseline_state = fit_baseline(TR, best_alpha)

pd.DataFrame(baseline_history).to_csv(
    RUN / "baseline_validation.csv", index=False
)
joblib.dump(baseline_state, RUN / "baseline.joblib")

# -------------------------------
# 4. 共通入力: source変化量PCA + 出発/目標条件
# -------------------------------
def make_inputs(fit):
    pair_rows = np.flatnonzero(
        np.isin(delta_data.target_ids, ids[fit])
    )
    pack = first.prepare(delta_data, pair_rows, pair_rows)

    expression = np.asarray([
        pack["source"][
            delta_data.source_ids[delta_data.target_ids == cid]
        ].numpy().mean(0)
        for cid in ids
    ], dtype=np.float64)

    conditions = np.asarray([
        pack["conditions"][
            delta_data.target_ids == cid
        ].numpy().mean(0)
        for cid in ids
    ], dtype=np.float64)

    count = min(
        SETTINGS["input_pca_max"],
        len(fit) - 1,
        expression.shape[1],
    )
    pca = PCA(
        n_components=count,
        svd_solver="randomized",
        random_state=42,
    ).fit(expression[fit])

    singular = pca.singular_values_
    keep = (
        singular > singular[0] * SETTINGS["input_pca_relative_tolerance"]
    )
    if not keep.any():
        raise ValueError("유효한 입력 PCA 축이 없습니다.")

    scores = pca.transform(expression)[:, keep]
    scaler = StandardScaler().fit(scores[fit])
    scores = scaler.transform(scores)

    X = np.c_[scores, conditions].astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("입력 NaN/Inf")

    state = dict(
        preprocess=pack["preprocess"],
        pca=pca, keep=keep, scaler=scaler,
    )
    return torch.tensor(X), state

with threadpool_limits(limits=SETTINGS["threads"]):
    inner_X, inner_prep = make_inputs(IT)
    final_X, final_prep = make_inputs(TR)

joblib.dump(inner_prep, RUN / "inner_preprocess.joblib")
joblib.dump(final_prep, RUN / "final_preprocess.joblib")

# -------------------------------
# 5. decoder
# latent node j = pathway j
# probeごとのfan-inで初期値を調整。符号は学習可能。
# -------------------------------
class DecoderModel(nn.Module):
    def __init__(self, input_dim, mode):
        super().__init__()
        probes, paths = membership.shape

        # 全条件で同じ共通encoder・小さい自由経路
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, SETTINGS["hidden"]),
            nn.SiLU(),
            nn.Dropout(SETTINGS["dropout"]),
        )
        self.latent = nn.Linear(SETTINGS["hidden"], paths)
        self.free_route = nn.Sequential(
            nn.Linear(
                SETTINGS["hidden"],
                SETTINGS["free_route_rank"],
                bias=False,
            ),
            nn.Linear(
                SETTINGS["free_route_rank"], probes, bias=False
            ),
        )
        self.bias = nn.Parameter(torch.zeros(probes))
        self.mode = mode

        if mode == "dense":
            self.dense = nn.Linear(paths, probes, bias=False)
        else:
            mask = membership if mode == "kegg" else random_membership
            edges = np.argwhere(mask > 0)
            self.register_buffer(
                "probe_index", torch.tensor(edges[:, 0], dtype=torch.long)
            )
            self.register_buffer(
                "pathway_index", torch.tensor(edges[:, 1], dtype=torch.long)
            )
            degree = mask.sum(1)
            initial = (
                torch.randn(len(edges))
                / torch.tensor(
                    np.sqrt(degree[edges[:, 0]]), dtype=torch.float32
                )
            )
            self.edge_weights = nn.Parameter(initial)

    def forward(self, x):
        hidden = self.encoder(x)
        latent = torch.nn.functional.silu(self.latent(hidden))

        if self.mode == "dense":
            output = self.dense(latent)
        else:
            values = (
                latent[:, self.pathway_index]
                * self.edge_weights[None, :]
            )
            output = x.new_zeros(len(x), membership.shape[0])
            output = output.index_add(1, self.probe_index, values)

        return output + self.free_route(hidden) + self.bias

def train_epoch(model, optimizer, X, target, fit):
    model.train()
    for local in torch.randperm(len(fit)).split(SETTINGS["batch_size"]):
        rows = fit[local.numpy()]
        prediction = model(X[rows])
        loss = (prediction - target[rows]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("학습 loss NaN/Inf")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

@torch.no_grad()
def predict_correction(model, X, rows, scale):
    model.eval()
    values = []
    for start in range(0, len(rows), SETTINGS["batch_size"]):
        batch = rows[start:start + SETTINGS["batch_size"]]
        values.append(model(X[batch]).numpy() * scale)
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError("보정 예측 NaN/Inf")
    return result

def optimizer_for(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=SETTINGS["learning_rate"],
        weight_decay=SETTINGS["weight_decay"],
    )

inner_residual = Y - inner_base
inner_scale = float(np.sqrt(np.mean(inner_residual[IT] ** 2)))
final_residual = Y - final_base
final_scale = float(np.sqrt(np.mean(final_residual[TR] ** 2)))
if min(inner_scale, final_scale) <= 1e-8:
    raise ValueError("학습 잔차 RMS가 너무 작습니다.")

inner_target = torch.tensor(
    inner_residual / inner_scale, dtype=torch.float32
)
final_target = torch.tensor(
    final_residual / final_scale, dtype=torch.float32
)

print("\n결과 폴더:", RUN, flush=True)
print("선택된 기준선 alpha:", best_alpha, flush=True)

# -------------------------------
# 6. 검증 선택 → 초기화 → 전체 훈련 → 테스트
# -------------------------------
records = []
modes = ["dense", "kegg", "random"]
total = len(modes) * len(SETTINGS["model_seeds"])

with threadpool_limits(limits=SETTINGS["threads"]):
    for seed in SETTINGS["model_seeds"]:
        for mode in modes:
            folder = RUN / f"{mode}_seed{seed}"
            folder.mkdir()

            first.seed_all(seed)
            model = DecoderModel(inner_X.shape[1], mode)
            optimizer = optimizer_for(model)
            n_parameters = sum(p.numel() for p in model.parameters())

            # 기본 예측만 사용하는 선택지도 처음부터 포함
            best = dict(
                epoch=0, strength=0.0,
                validation_response_rmse=rmse(Y[IV], inner_base[IV]),
            )
            history = [best.copy()]

            for epoch in range(1, SETTINGS["epochs"] + 1):
                train_epoch(
                    model, optimizer, inner_X, inner_target, IT
                )
                if epoch in SETTINGS["validation_epochs"]:
                    correction = predict_correction(
                        model, inner_X, IV, inner_scale
                    )
                    for strength in SETTINGS["strengths"]:
                        score = rmse(
                            Y[IV],
                            inner_base[IV] + strength * correction,
                        )
                        row = dict(
                            epoch=epoch, strength=strength,
                            validation_response_rmse=score,
                        )
                        history.append(row)
                        if score < best["validation_response_rmse"]:
                            best = row.copy()

                    pd.DataFrame(history).to_csv(
                        folder / "validation_history.csv", index=False
                    )
                    print(
                        f"{mode} seed={seed} epoch={epoch}: "
                        f"best val={best['validation_response_rmse']:.6f}, "
                        f"strength={best['strength']}",
                        flush=True,
                    )

            save_json(folder / "selection.json", best)
            prediction = final_base[TE].copy()

            if best["strength"] > 0:
                first.seed_all(seed)
                model = DecoderModel(final_X.shape[1], mode)
                optimizer = optimizer_for(model)

                for epoch in range(1, best["epoch"] + 1):
                    train_epoch(
                        model, optimizer,
                        final_X, final_target, TR,
                    )
                    if epoch == 1 or epoch % 5 == 0:
                        print(
                            f"재학습 {mode} seed={seed}: "
                            f"{epoch}/{best['epoch']}",
                            flush=True,
                        )

                prediction += best["strength"] * predict_correction(
                    model, final_X, TE, final_scale
                )
                torch.save(model.state_dict(), folder / "model.pt")

            mse = np.mean((Y[TE] - prediction) ** 2, axis=1)
            record = dict(
                decoder=mode,
                model_seed=seed,
                response_rmse=float(np.sqrt(mse.mean())),
                baseline_rmse=rmse(Y[TE], final_base[TE]),
                parameter_count=n_parameters,
                **best,
            )
            np.savez_compressed(
                folder / "predictions.npz",
                target_ids=ids[TE],
                prediction=prediction,
            )
            detail = meta.iloc[TE][
                ["condition_id", "compound", "time_h", "level"]
            ].copy()
            detail["response_mse"] = mse
            detail.to_csv(folder / "condition_metrics.csv", index=False)

            save_json(folder / "complete.json", record)
            records.append(record)
            pd.DataFrame(records).to_csv(
                RUN / "results.csv", index=False
            )
            print(
                f"완료 {len(records)}/{total}: "
                f"{mode} seed={seed}, "
                f"test={record['response_rmse']:.6f}",
                flush=True,
            )

# 保存するscaleは最終学習資料だけから計算
save_json(RUN / "final_settings.json", dict(
    baseline_alpha=best_alpha,
    residual_scale=final_scale,
    settings=SETTINGS,
))
np.savez_compressed(
    RUN / "baseline_predictions.npz",
    target_ids=ids[TE],
    prediction=final_base[TE],
)

results = pd.DataFrame(records)
summary = results.groupby("decoder").response_rmse.agg(
    ["mean", "std", "count"]
)
summary.to_csv(RUN / "summary.csv")

wide = results.pivot(
    index="model_seed", columns="decoder", values="response_rmse"
)
paired = pd.DataFrame({
    "kegg_minus_random": wide["kegg"] - wide["random"],
    "kegg_minus_dense": wide["kegg"] - wide["dense"],
})
paired.to_csv(RUN / "paired_differences.csv")
save_json(RUN / "completed.json", dict(completed_jobs=len(records)))

print("\n전체 완료:", RUN)
print("목표 조건 Ridge RMSE:", rmse(Y[TE], final_base[TE]))
display(summary)
display(paired)