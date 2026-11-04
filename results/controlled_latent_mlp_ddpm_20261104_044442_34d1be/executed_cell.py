# 방금 pathway decoder 실험을 완료한 같은 커널에서 실행
# 구조 포함 입력을 고정하고 latent MLP vs x0-DDPM 비교
# 기존 결과 보존 / CPU / GitHub push 없음

required = [
    "inner_X", "final_X", "inner_base", "final_base",
    "data", "Y", "ids", "meta", "IT", "IV", "TR", "TE",
    "RUN", "first", "save_json", "inner_prep", "final_prep",
    "baseline_state", "best_alpha",
]
missing = [k for k in required if k not in globals()]
if missing:
    raise RuntimeError(
        "pathway decoder 실험을 완료한 커널에서 실행해 주세요: "
        + ", ".join(missing)
    )

from pathlib import Path
from datetime import datetime
import uuid
import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
import joblib
from threadpoolctl import threadpool_limits

PARENT_DECODER = Path(RUN)
COMPARE = PARENT_DECODER.parent / (
    "controlled_latent_mlp_ddpm_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_" + uuid.uuid4().hex[:6]
)
COMPARE.mkdir(exist_ok=False)

PLAN = dict(
    seed=42,
    models=["mlp", "ddpm"],
    output_components_max=32,
    epochs=100,
    validation_epochs=[1, 25, 50, 75, 100],
    learning_rate=0.0003,
    weight_decay=0.01,
    batch_size=64,
    steps=30,
    samples=16,
    strengths=[0.0, 0.1, 0.25, 0.5, 1.0],
    threads=4,
    diffusion_prediction="x0",
    schedule="cosine",
    structure=True,
    target="PCA_of_training_baseline_residual",
    residual_policy="in_sample_training_baseline_residual",
    exploratory=True,
)
save_json(COMPARE / "config.json", PLAN)
(COMPARE / "executed_cell.py").write_text(
    get_ipython().history_manager.input_hist_raw[-1],
    encoding="utf-8",
)

# 入力・分割・前処理の根拠も保存
for name in [
    "config.json", "final_settings.json",
    "parent_run.json", "parent_compound_split.csv",
    "source_delta_audit.csv", "baseline_validation.csv",
]:
    if (PARENT_DECODER / name).is_file():
        shutil.copy2(
            PARENT_DECODER / name, COMPARE / ("parent_" + name)
        )
joblib.dump(inner_prep, COMPARE / "inner_input_preprocess.joblib")
joblib.dump(final_prep, COMPARE / "final_input_preprocess.joblib")
joblib.dump(baseline_state, COMPARE / "baseline.joblib")
save_json(COMPARE / "parent.json", dict(
    folder=str(PARENT_DECODER),
    baseline_alpha=float(best_alpha),
))

# 共通入力に同じMorgan+presenceを追加
structure = torch.tensor(
    data.structure[
        [data.compound_to_index[c] for c in meta.compound]
    ],
    dtype=torch.float32,
)
XI = torch.cat([inner_X, structure], dim=1)
XF = torch.cat([final_X, structure], dim=1)

def latent_prepare(fit, baseline_prediction):
    residual = Y - baseline_prediction
    n = min(PLAN["output_components_max"], len(fit) - 1)
    pca = PCA(
        n_components=n,
        svd_solver="randomized",
        random_state=PLAN["seed"],
    ).fit(residual[fit])

    keep = (
        pca.singular_values_
        > pca.singular_values_[0] * 1e-8
    )
    if not keep.any():
        raise ValueError("有効な残差PCA軸がありません。")

    scores = pca.transform(residual)[:, keep]

    # 共通スカラーで調整。軸ごとにwhiteningしない。
    scale = float(np.sqrt(np.mean(scores[fit] ** 2)))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("残差latent RMSが不正です。")

    return (
        torch.tensor(scores / scale, dtype=torch.float32),
        dict(pca=pca, keep=keep, scale=scale),
    )

def restore_latent(scores, state):
    # 除いたPCA軸は0。平均残差はPCA inverse相当で復元。
    components = state["pca"].components_[state["keep"]]
    return (
        scores * state["scale"] @ components
        + state["pca"].mean_
    )

ZI, latent_inner = latent_prepare(IT, inner_base)
ZF, latent_final = latent_prepare(TR, final_base)
joblib.dump(latent_inner, COMPARE / "inner_output_preprocess.joblib")
joblib.dump(latent_final, COMPARE / "final_output_preprocess.joblib")

# 同じnetworkを使用。
# MLPはnoisy latent=0・time=0を固定して直接予測。
# DDPMはnoisy latentとtimeを使用してx0予測。
class LatentPredictor(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.context = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
        )
        self.time = nn.Embedding(PLAN["steps"], 16)
        self.head = nn.Sequential(
            nn.Linear(64 + latent_dim + 16, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x, noisy, t):
        return self.head(torch.cat([
            self.context(x), noisy, self.time(t)
        ], dim=1))

# cosine schedule
T = PLAN["steps"]
u = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
f = torch.cos((u + 0.008) / 1.008 * torch.pi / 2).square()
cumulative = f / f[0]
beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-6, .999).float()
alpha = 1 - beta
abar = torch.cumprod(alpha, dim=0)
abar_prev = torch.cat([torch.ones(1), abar[:-1]])

def train_epoch(kind, model, optimizer, X, Z, fit):
    model.train()
    for local in torch.randperm(len(fit)).split(PLAN["batch_size"]):
        rows = fit[local.numpy()]
        truth = Z[rows]
        if kind == "mlp":
            t = torch.zeros(len(rows), dtype=torch.long)
            noisy = torch.zeros_like(truth)
        else:
            t = torch.randint(T, (len(rows),))
            noise = torch.randn_like(truth)
            noisy = (
                abar[t, None].sqrt() * truth
                + (1 - abar[t, None]).sqrt() * noise
            )
        prediction = model(X[rows], noisy, t)
        loss = (prediction - truth).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

@torch.no_grad()
def predict_scores(kind, model, X, rows, seed):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    outputs = []

    for start in range(0, len(rows), PLAN["batch_size"]):
        batch = rows[start:start + PLAN["batch_size"]]
        context = X[batch]

        if kind == "mlp":
            noisy = torch.zeros(len(batch), model.latent_dim)
            t = torch.zeros(len(batch), dtype=torch.long)
            result = model(context, noisy, t)
        else:
            context = context.repeat_interleave(PLAN["samples"], dim=0)
            noisy = torch.randn(
                (len(context), model.latent_dim), generator=generator
            )
            for step in reversed(range(T)):
                t = torch.full(
                    (len(context),), step, dtype=torch.long
                )
                x0 = model(context, noisy, t)
                denominator = 1 - abar[step]

                # DDPM posterior mean using predicted x0
                mean = (
                    beta[step] * abar_prev[step].sqrt()
                    / denominator * x0
                    + alpha[step].sqrt()
                    * (1 - abar_prev[step])
                    / denominator * noisy
                )
                noisy = mean
                if step:
                    variance = (
                        beta[step] * (1 - abar_prev[step])
                        / denominator
                    )
                    noisy = noisy + variance.sqrt() * torch.randn(
                        noisy.shape, generator=generator
                    )
            result = noisy.reshape(
                len(batch), PLAN["samples"], model.latent_dim
            ).mean(dim=1)

        array = result.numpy()
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite latent predictions")
        outputs.append(array)

    return np.concatenate(outputs)

def rmse(actual, prediction):
    return float(np.sqrt(np.mean((actual - prediction) ** 2)))

def make_model(X, Z):
    first.seed_all(PLAN["seed"])
    model = LatentPredictor(X.shape[1], Z.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PLAN["learning_rate"],
        weight_decay=PLAN["weight_decay"],
    )
    return model, optimizer

torch.set_num_threads(PLAN["threads"])
records = [dict(
    model="target_conditions_ridge",
    response_rmse=rmse(Y[TE], final_base[TE]),
)]
print("결과 폴더:", COMPARE, flush=True)

with threadpool_limits(limits=PLAN["threads"]):
    for kind in PLAN["models"]:
        folder = COMPARE / kind
        folder.mkdir()

        model, optimizer = make_model(XI, ZI)
        best = dict(
            epoch=0,
            strength=0.0,
            validation_response_rmse=rmse(Y[IV], inner_base[IV]),
        )
        history = [best.copy()]

        for epoch in range(1, PLAN["epochs"] + 1):
            train_epoch(kind, model, optimizer, XI, ZI, IT)

            if epoch in PLAN["validation_epochs"]:
                scores = predict_scores(
                    kind, model, XI, IV, seed=20042
                )
                correction = restore_latent(scores, latent_inner)

                for strength in PLAN["strengths"]:
                    score = rmse(
                        Y[IV], inner_base[IV] + strength * correction
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
                    f"{kind} epoch={epoch}: "
                    f"best val={best['validation_response_rmse']:.6f}, "
                    f"strength={best['strength']}",
                    flush=True,
                )

        save_json(folder / "selection.json", best)
        prediction = final_base[TE].copy()

        if best["strength"] > 0:
            model, optimizer = make_model(XF, ZF)

            for epoch in range(1, best["epoch"] + 1):
                train_epoch(kind, model, optimizer, XF, ZF, TR)
                if epoch == 1 or epoch % 10 == 0 or epoch == best["epoch"]:
                    print(
                        f"재학습 {kind}: {epoch}/{best['epoch']}",
                        flush=True,
                    )

            scores = predict_scores(
                kind, model, XF, TE, seed=30042
            )
            correction = restore_latent(scores, latent_final)
            prediction += best["strength"] * correction
            torch.save(model.state_dict(), folder / "model.pt")

        mse = np.mean((Y[TE] - prediction) ** 2, axis=1)
        record = dict(
            model=kind,
            response_rmse=float(np.sqrt(mse.mean())),
            **best,
        )
        records.append(record)

        np.savez_compressed(
            folder / "predictions.npz",
            target_ids=ids[TE], prediction=prediction,
        )
        detail = meta.iloc[TE][
            ["condition_id", "compound", "time_h", "level"]
        ].copy()
        detail["response_mse"] = mse
        detail.to_csv(folder / "condition_metrics.csv", index=False)
        save_json(folder / "complete.json", record)

        pd.DataFrame(records).to_csv(
            COMPARE / "results.csv", index=False
        )
        print("완료:", record, flush=True)

np.savez_compressed(
    COMPARE / "baseline_predictions.npz",
    target_ids=ids[TE], prediction=final_base[TE],
)
save_json(COMPARE / "completed.json", dict(completed_models=2))

print("\n전체 완료:", COMPARE)
display(pd.DataFrame(records))