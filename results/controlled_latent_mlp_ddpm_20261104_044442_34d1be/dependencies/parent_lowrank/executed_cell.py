# 독립 실행 셀: 저차원 반응 Ridge + 입력 표현 비교
# 기존 데이터 읽기만 / 새 결과 폴더 저장 / GitHub push 없음
# CPU 실행. 필요한 패키지:
# numpy pandas scipy scikit-learn torch joblib threadpoolctl

from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import subprocess
import tempfile
import sys
import json
import hashlib
import shutil
import platform
import uuid

import numpy as np
import pandas as pd
import torch
import joblib
import sklearn
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

BASE = Path("/home/seowon1211/data1/conditioned_expression")
PREPARED = BASE / "prepared_controls"
STRUCTURE = BASE / "morgan_matched/morgan_binary.csv"
ANNOTATION = BASE / "kegg_rat_20261102_034010_265aed"

CONFIG = dict(
    split_seed=42,
    output_ranks=[8, 16, 32],
    ridge_alphas=[10.0, 100.0, 1000.0],
    representations=["C", "E", "E_S", "P_S", "PCA_S"],
    threads=4,
    target="vivo_treated_minus_matched_control",
    method="baseline_plus_PCA_response_ridge",
    selection="validation_response_rmse",
    exploratory=True,
)

def save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

# 기존 코드 혼용 방지
for name in [
    "first_structure_mlp",
    "pathway_structure_mlp",
    "repeated_pathway_validation",
]:
    if name in sys.modules:
        raise RuntimeError(
            "새 커널에서 실행해 주세요. 이미 로드된 모듈: " + name
        )

files = [
    PREPARED / name for name in [
        "vitro_expression.npy", "vivo_expression.npy",
        "vitro_conditions.csv", "vivo_conditions.csv",
        "pairs.csv.gz", "probes.csv",
    ]
] + [
    STRUCTURE,
    ANNOTATION / "projection.npy",
    ANNOTATION / "pathway_audit.csv",
    ANNOTATION / "provenance.json",
]
for p in files:
    if not p.is_file():
        raise FileNotFoundError(p)

# 로더와 KEGG 정합성 검사만 기존 고정 커밋에서 사용
REPO = Path(tempfile.mkdtemp(prefix="ivive_lowrank_code_"))
subprocess.run(
    ["git", "clone", "https://github.com/kiseowon/practice.git", str(REPO)],
    check=True,
)
subprocess.run(
    ["git", "-C", str(REPO), "checkout", "--detach", "3300ade"],
    check=True,
)
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import first_structure_mlp as first
import pathway_structure_mlp as pathway
import repeated_pathway_validation as repeated

OUT = BASE / (
    "lowrank_pathway_pilot_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_" + uuid.uuid4().hex[:6]
)
OUT.mkdir(exist_ok=False)

source_files = [
    SCRIPTS / name for name in [
        "first_structure_mlp.py",
        "pathway_structure_mlp.py",
        "repeated_pathway_validation.py",
    ]
]
save_json(OUT / "run.json", dict(
    config=CONFIG,
    helper_commit=subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip(),
    input_sha256={str(p): digest(p) for p in files},
    helper_sha256={p.name: digest(p) for p in source_files},
    versions=dict(
        python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__,
        sklearn=sklearn.__version__, torch=torch.__version__,
    ),
))
(OUT / "executed_cell.py").write_text(
    get_ipython().history_manager.input_hist_raw[-1],
    encoding="utf-8",
)
for p in source_files:
    shutil.copy2(p, OUT / p.name)

data = first.Data(SimpleNamespace(
    prepared=PREPARED, structure=STRUCTURE
))
projection, _, provenance = pathway.load_projection(
    ANNOTATION, PREPARED
)
data.structure_audit.to_csv(OUT / "structure_audit.csv", index=False)

# 1. 실제 vivo 처리군−대조군 반응
vo = data.vo
controls = vo.loc[vo.level.eq("Control")]
keys = ["compound", "unit", "time_h"]
if controls.duplicated(keys).any():
    raise ValueError("대조군 연결이 모호합니다.")
control_map = controls.set_index(keys).condition_id.to_dict()

ids = np.unique(
    data.target_ids[
        ~data.pairs.level_target.eq("Control").to_numpy()
    ]
)
meta = vo.iloc[ids].reset_index(drop=True)
responses, audits = [], []
for cid in ids:
    row = vo.iloc[cid]
    key = (row.compound, row.unit, row.time_h)
    if key not in control_map:
        raise ValueError(f"대조군 없음: {key}")
    control_id = int(control_map[key])
    responses.append(data.target[cid] - data.target[control_id])
    audits.append(dict(
        condition_id=int(cid), control_condition_id=control_id,
        compound=row.compound, time_h=row.time_h,
        unit=row.unit, level=row.level,
    ))

Y = np.asarray(responses, dtype=np.float64)
pd.DataFrame(audits).to_csv(
    OUT / "response_target_audit.csv", index=False
)

# 2. 화합물 단위 내부 학습 / 검증 / 테스트
split = repeated.compound_split(data.names, CONFIG["split_seed"])
split.to_csv(OUT / "compound_split.csv", index=False)

def positions(stage):
    names = split.loc[split.stage.eq(stage), "compound"]
    return np.flatnonzero(meta.compound.isin(names))

IT = positions("inner_train")
IV = positions("inner_validation")
TE = positions("outer_test")
TR = np.sort(np.r_[IT, IV])

def baseline(fit):
    mapping = {}
    for key, group in meta.iloc[fit].groupby(["time_h", "level"]):
        vectors = [
            Y[g.index.to_numpy()].mean(0)
            for _, g in group.groupby("compound")
        ]
        mapping[key] = np.mean(vectors, axis=0)

    # 모든 위치로 변환하되, 평균 계산에는 fit만 사용
    result = []
    for _, row in meta.iterrows():
        key = (row.time_h, row.level)
        if key not in mapping:
            raise ValueError(f"학습에 없는 시간×등급: {key}")
        result.append(mapping[key])
    return np.asarray(result), mapping

def aggregate_pair_values(values):
    """각 target에 연결된 source/pair 특징을 평균."""
    return np.asarray([
        values[data.target_ids == cid].mean(0) for cid in ids
    ], dtype=np.float64)

def prepare_inputs(fit):
    pair_rows = np.flatnonzero(
        np.isin(data.target_ids, ids[fit])
    )
    # source scaler와 조건 encoder는 fit 자료에만 적합
    pack = first.prepare(data, pair_rows, pair_rows)

    E = aggregate_pair_values(
        pack["source"][data.source_ids].numpy()
    )
    C = aggregate_pair_values(pack["conditions"].numpy())

    index = [data.compound_to_index[c] for c in meta.compound]
    S = data.structure[index].astype(np.float64)

    # 기존 projection의 고정 gene-balanced 평균 가중치 사용
    P = E @ projection
    pscale = StandardScaler().fit(P[fit])
    P = pscale.transform(P)

    # KEGG pathway와 같은 명목 차원으로 PCA 입력 대조
    input_rank = min(
        projection.shape[1], len(fit) - 1, E.shape[1]
    )
    pca = PCA(
        n_components=input_rank,
        svd_solver="randomized",
        random_state=42,
    ).fit(E[fit])
    PC = pca.transform(E)
    pcscale = StandardScaler().fit(PC[fit])
    PC = pcscale.transform(PC)

    X = {
        "C": C,
        "E": np.c_[C, E],
        "E_S": np.c_[C, E, S],
        "P_S": np.c_[C, P, S],
        "PCA_S": np.c_[C, PC, S],
    }
    state = dict(
        original_preprocess=pack["preprocess"],
        pathway_scaler=pscale,
        input_pca=pca,
        input_pca_scaler=pcscale,
        input_pca_rank=input_rank,
    )
    return X, state

def response_basis(fit, B):
    # 평균 기준선에서 벗어난 반응을 학습 자료에서만 압축
    R = Y[fit] - B[fit]
    mean = R.mean(0)
    max_rank = min(max(CONFIG["output_ranks"]), len(fit) - 1)
    pca = PCA(
        n_components=max_rank,
        svd_solver="randomized",
        random_state=42,
    ).fit(R - mean)
    Z = pca.transform(R - mean)
    return mean, pca, Z

def kernel_prepare(X, fit):
    # primal 고차원 역행렬 대신 target 수 크기의 dual 계산
    mean = X[fit].mean(0)
    A = X[fit] - mean
    eigenvalues, eigenvectors = eigh(A @ A.T)
    eigenvalues = np.maximum(eigenvalues, 0)
    return mean, A, eigenvalues, eigenvectors

def predict_lowrank(X, kernel, pca, Z, response_mean, B, rank, alpha):
    mean, A, eigenvalues, eigenvectors = kernel
    dual = eigenvectors @ (
        (eigenvectors.T @ Z[:, :rank])
        / (eigenvalues[:, None] + alpha)
    )
    scores = (X - mean) @ A.T @ dual
    pred = B + response_mean + scores @ pca.components_[:rank]
    if not np.isfinite(pred).all():
        raise ValueError("예측 NaN/Inf")
    return pred, dual

def rmse(actual, prediction):
    return float(np.sqrt(np.mean((actual - prediction) ** 2)))

history, selections = [], {}
with threadpool_limits(limits=CONFIG["threads"]):
    # 3. 내부 검증에서만 표현별 설정 선택
    inner_X, inner_state = prepare_inputs(IT)
    inner_B, _ = baseline(IT)
    rmean, target_pca, Z = response_basis(IT, inner_B)

    print("결과 폴더:", OUT, flush=True)
    print("화합물 분할:\n", split.groupby("stage").size(), flush=True)

    for arm, X in inner_X.items():
        kernel = kernel_prepare(X, IT)
        best = None
        for rank in CONFIG["output_ranks"]:
            if rank > Z.shape[1]:
                continue
            for alpha in CONFIG["ridge_alphas"]:
                pred, _ = predict_lowrank(
                    X, kernel, target_pca, Z, rmean,
                    inner_B, rank, alpha,
                )
                score = rmse(Y[IV], pred[IV])
                row = dict(
                    arm=arm, rank=rank, alpha=alpha,
                    validation_response_rmse=score,
                )
                history.append(row)
                if best is None or score < best["validation_response_rmse"]:
                    best = row.copy()

        selections[arm] = best
        pd.DataFrame(history).to_csv(
            OUT / "validation_history.csv", index=False
        )
        print("검증 선택:", best, flush=True)

    save_json(OUT / "selection.json", selections)

    # 4. 설정 고정 후 전체 훈련 화합물로 다시 적합
    final_X, final_state = prepare_inputs(TR)
    final_B, baseline_mapping = baseline(TR)
    rmean, target_pca, Z = response_basis(TR, final_B)
    joblib.dump(final_state, OUT / "feature_preprocess.joblib")

    records = []
    detail = meta.iloc[TE][
        ["condition_id", "compound", "time_h", "unit", "level"]
    ].copy()

    for name, pred in [
        ("zero_response", np.zeros_like(Y)),
        ("time_level_mean", final_B),
    ]:
        mse = np.mean((Y[TE] - pred[TE]) ** 2, axis=1)
        detail[name + "_mse"] = mse
        records.append(dict(
            method=name,
            response_rmse=float(np.sqrt(mse.mean())),
        ))

    for arm, X in final_X.items():
        selected = selections[arm]
        kernel = kernel_prepare(X, TR)
        prediction, dual = predict_lowrank(
            X, kernel, target_pca, Z, rmean,
            final_B, selected["rank"], selected["alpha"],
        )
        mse = np.mean((Y[TE] - prediction[TE]) ** 2, axis=1)
        detail[arm + "_mse"] = mse
        score = float(np.sqrt(mse.mean()))

        records.append(dict(
            method=arm,
            response_rmse=score,
            rank=selected["rank"],
            alpha=selected["alpha"],
        ))
        np.savez_compressed(
            OUT / f"{arm}_predictions.npz",
            target_ids=ids[TE], prediction=prediction[TE],
        )
        joblib.dump(dict(
            representation=arm,
            feature_mean=kernel[0],
            training_features_centered=kernel[1],
            dual=dual,
            response_mean=rmean,
            output_components=target_pca.components_[:selected["rank"]],
            baseline_mapping=baseline_mapping,
            selection=selected,
        ), OUT / f"{arm}_model.joblib")
        print(f"완료: {arm}, test RMSE={score:.6f}", flush=True)

table = pd.DataFrame(records)
base_rmse = table.loc[
    table.method.eq("time_level_mean"), "response_rmse"
].iloc[0]
table["delta_vs_time_level_mean"] = table.response_rmse - base_rmse

table.to_csv(OUT / "results.csv", index=False)
detail.to_csv(OUT / "condition_metrics.csv", index=False)
save_json(OUT / "completed.json", dict(
    completed=True,
    n_models=len(CONFIG["representations"]),
))

print("\n전체 완료:", OUT)
display(table.sort_values("response_rmse"))