# 코드 학습과 연구미팅 PPT 초안

작성 기준: 2026-11-05. 기본 예측 모델과 파라미터 선택 코드를 구분합니다. 현재 진행 중인 조합 탐색의 최종 결과는 아직 전달되지 않았으므로 완료된 결과처럼 기재하지 않습니다. 아래는 슬라이드에 넣을 문장과 설명 노트이며 실제 PPTX 파일은 아닙니다.

## 슬라이드 1 — 연구 질문과 현재 비교 범위

**슬라이드 문장**

- 미관측 화합물의 in vivo 전사체 반응을 in vitro 반응과 화학 구조로 예측할 수 있는가?
- 공통 잔차 예측 틀에서 MLP·cGAN·x₀-DDPM을 비교한다.
- 구조 on/off 대조로 Morgan 입력의 기여를 확인한다.

**발표 노트**

생물학적 정보를 사전 학습 과정에 활용한다는 큰 연구 질문에서 시작했으나, 현재 반복 실험은 KEGG가 없는 구조·발현 기반 모델 비교다. KEGG 실험은 별도 탐색으로 구분한다. 이번 결과를 pathway 효과 검증이나 AIVIVE 재현으로 표현하지 않는다.

## 슬라이드 2 — 데이터와 상위 전처리

**슬라이드 문장**

- Rat liver, 140개 화합물, 고정된3,453개 probe.
- AIVIVE의 RMA 정규화 Zenodo 발현 자료를 사용한다.
- 화합물·용량·단위·시간·등급별 replicate 평균과 매칭 대조군 차감을 수행한다.
- 정답은 개별 동물 분포가 아닌 조건 평균 반응이다.

**발표 노트**

사용자 서버에서 로컬 vitro/vivo_train_test.csv의 MD5가 배포 파일과 일치했다. 우리가 RMA를 새로 실행한 것은 아니다. RMA와 학습용 표준화는 다른 단계다. 변화량은 동일 compound·unit·time_h의 실제 대조군을 뺀다. 원자료 RMA의 모든 옵션을 재현했다거나 corrected metadata의 과거 생성 버전까지 해결됐다는 뜻은 아니다.

출처: [발현 자료 확인](expression-source.md).

## 슬라이드 3 — 입력과 예측 단위

**슬라이드 문장**

| 입력 | 표현 |
| --- | --- |
| in vitro 반응 | 표준화 후 target별 source 평균, PCA 최대64차원 |
| 구조 | Morgan2,048bit + presence1, 별도 PCA 없음 |
| 조건 | source/target 각각 dose·time·unit·level |
| 출력 | 목표 vivo의3,453probe 처리군−대조군 반응 |

**발표 노트**

화합물별 여러 in vitro 조건을 먼저 요약한다. 한 농도·시간의 vitro 입력 하나를 독립적으로 번역한 뒤 예측을 평균하는 실험과 다르다. 새로운 화합물에서도 해당 source 조건 묶음을 확보할 수 있다는 적용 전제가 있다. source 스케일러는 고유 학습 source에 적합하고, source/target 조건의 dose는 log1p, dose/time은 표준화, unit/level은 one-hot이다. 서로 다른 단위를 물리적으로 환산하지 않았다.

PCA의 source 평균 특징은 target마다 행을 가지므로 같은 화합물의 특징이 반복될 수 있다. target 행 수가 독립 화합물 수를 늘리는 것은 아니다. 구조 누락3개는0+presence0으로 처리한다. 화합물 이름은 매칭용이지 학습 특징 자체가 아니다.

## 슬라이드 4 — 기준선과 잔차 학습

**슬라이드 문장**

- 변화 없음: 모든 probe의 예측 반응0.
- 시간×등급 평균: 훈련 화합물의 공통 반응.
- 목표 조건 Ridge: 세 잔차 모델이 공유하는 기본 예측.
- 최종 반응 = Ridge 예측 + λ × 복원한 잔차.

```mermaid
flowchart LR
    T[목표 vivo 조건] --> B[공통 Ridge]
    X[vitro 반응·구조·조건] --> M[잠재 잔차 예측기]
    M --> D[고정 PCA로 잔차 복원]
    D --> W[검증으로 선택한 λ]
    B --> A[더하기]
    W --> A
    A --> Y[vivo 반응 예측]
```

**발표 노트**

예를 들어 훈련 화합물A/B의24시간·High 반응이+0.4/+0.8이면 해당 probe의 조건 평균 기준선은+0.6이다. 실제로는3,453probe 벡터이며 한 화합물에 여러 조건이 있으면 화합물 내 평균 후 화합물 간 평균을 낸다. High는 같은 물리적 투여량이 아니다.

Ridge에는 목표 vivo 조건만 입력한다. 모델이 보정할 잔차는 실제 반응−훈련 내 Ridge 예측이다. OOF 잔차가 아니며 이를 외부 test 누수와 혼동하지 않는다. 공통 Ridge가 포함된 모델임을 설명하면서 그 기여를 숨기지 않는다.

출처: [기준선 설명](baselines.md).

## 슬라이드 5 — 입력 PCA와 출력 PCA는 다르다

**슬라이드 문장**

- 입력 PCA: in vitro 반응을 요약한다(최대64).
- 출력 PCA: Ridge가 놓친 vivo 잔차를 요약한다(최대32).
- PCA와 scaler는 각 학습 집합에만 적합한다.
- 거의0인 PCA축은 제거하고, 출력 score는 공통 RMS 한 개로 조정한다.

**발표 노트**

출력은 PCA가 제거한 평균 잔차를 포함해 복원하고, 복원 보정 전체에λ를 곱한다. λ0이면 Ridge만 사용한다. 잠재 공간의 MSE가 모든 probe의 모든 반응을 직접 학습하는 것과 같지는 않다. PCA 밖의 반응 성분은 예측할 수 없다. 고정 PCA 복원과 별도 학습 decoder를 구분한다. 이 실험은 AE·VAE를 쓰지 않는다.

코드: [prepare_stage·restore](../scripts/repeated_latent_validation.py).

## 슬라이드 6 — 작은 조건부 x₀-DDPM

**슬라이드 문장**

```text
발현PCA + Morgan + 실험조건 → Linear+SiLU → context64
                                              +
                        noisy latent 최대32 + time embedding16
                                              ↓
                                Linear+SiLU → hidden128
                                              ↓
                                       latent x₀ 예측
```

- 학습: timestep과 Gaussian noise를 뽑고 깨끗한 latent x₀의 MSE를 최소화.
- 예측: cosine30단계 역과정,16회 생성 평균.
- 최종값: PCA 역변환한 잔차를 약하게 더한다.

**발표 노트**

U-Net이 아닌 작은 완전연결 denoiser다. 현재 구현에는 identity skip이나 dropout이 없고 epsilon이 아니라x₀를 예측한다.30단계는30층이 아니라 같은 network를t에 따라 반복 적용한다는 뜻이다. 앞서 보존한 다른 epsilon-DDPM 구현과 섞어 설명하지 않는다. 생성 변동을 실제 동물 간 분포나 calibration으로 해석하지 않는다.

MLP는 같은 형태의 network에서 noisy/time을0으로 고정해 잠재값을 직접 예측한다. cGAN은 독립적인 generator/discriminator context를 쓰는 순수 조건부 BCE 버전이고 재구성 MSE는 없다. 같은 데이터·선택 예산이지만 모든 family의 parameter·연산량이 같거나 충분히 최적화됐다는 뜻은 아니다.

코드: [공통 모델·학습·생성](../scripts/repeated_latent_validation.py).

## 슬라이드 7 — 분할·선택·재학습

**슬라이드 문장**

- 화합물5-fold, split seed42.
- 각 fold: 내부 학습89 / 검증23 / 외부 시험28.
- 검증으로 Ridge α·epoch·보정강도를 선택.
- 선택 후112개에 전처리부터 다시 적합하고 모델을 초기화해 재학습.
- 모델seed42/43/44: 세 family 합계45조건.

**발표 노트**

분할 seed를3번 반복한5-fold는 아니다.5-fold 한 묶음에 초기화3개다. 시험 화합물의 actual control은 정답 계산에만 사용한다. 동일 target이 source pair에 반복돼도 예측/지표는 고유 target에 대해 계산한다. pooled RMSE는 fold RMSE의 단순 평균이 아니다. target·probe 균등 가중이며 compound macro metric이 아니다.

연구 전체에서 같은 데이터를 관찰해 설계를 발전시켰으므로 독립 확증 시험이라고 부르지 않는다.3seed를 독립 화합물 시험3개로 세지 않는다.

## 슬라이드 8 — 완료 결과: 작은 구조 입력 효과

**슬라이드 문장**

| 모델 | 구조off 평균 RMSE | 구조on 평균 RMSE | on이 낮은 seed |
| --- | ---: | ---: | ---: |
| MLP | 0.291572 | 0.290797 | 3/3 |
| DDPM | 0.291770 | 0.290991 | 2/3 |
| cGAN | 0.294571 | 0.294099 | 2/3 |

- 구조를 제공했을 때 평균 RMSE가 소폭 낮았다.
- DDPM은 현재cGAN보다 낮았지만MLP 대비 일관된 우월성은 없었다.
- 시간×등급 평균0.288403은 넘지 못했다.

**발표 노트**

구조 효과의 가능성을 보여주는 동일조건 ablation이다. 모든 지표나 seed에서 개선된 것은 아니다. 약0.27%의 RMSE감소는 큰 효과라고 과장하지 않는다. cGAN은 특정한 기본 구현이며 AIVIVE 재현이 아니다. 단독checkpoint/예측을 받은 과거 단일분할 백업과 최신45조건 로그 보고를 구분한다.

출처: [구조 대조의 서버 로그와 표](../results/structure_ablation_20261105_020903_dfd7fc/README.md).

## 슬라이드 9 — 파라미터 근거와 진행 중인 실험

**슬라이드 문장**

- 초기64/128은 작은 토이모델의 고정값이며 최적성 근거는 없었다.
- width5후보×3seed에서 context64·hidden64가 가장 낮은 내부 검증 RMSE를 보였다.
- 현재 hidden2개×학습률3개×weight decay2개=12조합을 검증 중이다.
- seed42 screen → 상위2개+reference에 seed43/44 추가: 총16~18fit.
- 외부 test 평가와 최종 refit은 하지 않는다.

**발표 노트**

완료된 width 실험의 reference 검증 RMSE는0.302810, hidden64는0.302517이다. 하나의 검증집합에서의 차이이며0.29대 OOF test 결과와 직접 비교하지 않는다. 현재 조합 탐색은 결과 미확보 상태로 표시한다. learning rate와 weight decay는 optimizer에 실제 적용하고, checkpoint·epoch·λ는 validation으로 고른다. 기존 구조 on/off는64/128을 그대로 유지해 사후 튜닝을 한쪽에 끼워 넣지 않았다.

각 숫자의 최고값을 따로 모아 합치는 것이 아니라 조합 자체를 평가한다. 최종 선택은 세 seed를 가진 shortlist 사이에서만 한다. screen 탈락 후보까지 모두3seed로 평가한 것은 아니다. 같은 검증 자료에 반복 선택한 값이므로 독립 성능 추정으로 해석하지 않는다.

코드: [candidates·shortlist·execute](../scripts/tune_ddpm_joint.py), [model_for·fit_trial](../scripts/validate_ddpm_architecture.py).

## 슬라이드 10 — 성능 지표와 결론의 한계

**슬라이드 문장**

- 주지표: 원래 반응 공간 RMSE. MAE·상관·방향 accuracy·Macro F1은 보조.
- 전체 accuracy94~98%는 변화 작음 다수 class의 영향을 크게 받는다.
- 부호 accuracy가 반응 크기 정확도를 뜻하지는 않는다.
- 구조 입력과 Diffusion 보정에 탐색적 신호가 있으나 강한 기준선 이상의 일반적 우월성은 미확인이다.

**발표 노트**

진폭τ0.5/1은 통계적 DEG나 검증된2배 변화 기준이 아니다. 실제+1.2를+0.01로 예측하면 부호는 맞지만 up 분류는 놓친다. 측정값은 조건 평균이며 개별 동물 분포 품질을 평가하지 않았다. cGAN 대비 차이를 AIVIVE 대비 성능 향상으로 표현하지 않는다. 향후 튜닝 모델의 일반화 성능은 각 outer fold 안에서 설정을 선택하는 nested 절차로 평가해야 한다.

## 코드 공부 순서와 핵심 Python 문법

| 순서 | 함수 | 읽을 내용 |
| --- | --- | --- |
| 1 | core.load_data | 배열·조건표·대조군 연결과 실제 반응Y |
| 2 | core.baseline_stage | 목표 조건만으로 Ridge 기본 예측 |
| 3 | core.prepare_stage | 학습 전용 scaler/PCA와 tensor X/Z |
| 4 | width.model_for | context·time·hidden 구조와 optimizer |
| 5 | core.train_epoch | noise 추가·x₀ MSE·backprop |
| 6 | core.predict_scores,core.restore | 역과정·16sample 평균·probe 공간 복원 |
| 7 | width.fit_trial | 검증 epoch/λ 선택과 best checkpoint |
| 8 | joint.candidates,shortlist | 12조합과 후보 제한 |
| 9 | joint.execute | 데이터 계약·학습 loop·초기화 반복·기록 |

- `itertools.product`는 각 후보값의 모든 조합을 만든다.
- `torch.cat(..., dim=1)`은 같은 batch의 특징 열을 이어 붙인다.
- `nn.Linear(a,b)`는 a개 입력을 b개 특징으로 변환하는 가중치를 학습한다.
- `loss.backward()`는 gradient를 계산하고 `optimizer.step()`은 가중치를 갱신한다.
- validation 정답은 optimizer 갱신에 직접 쓰지 않고 설정 선택에 사용한다.

로그 `epoch=75: {'epoch':25,...}`는 현재75epoch까지 학습했지만 최고 검증 checkpoint는25epoch라는 뜻이다. 현재 코드는 early stopping으로25에서 종료하지 않고100epoch의 같은 예산에서 최선 설정을 기록한다. checkpoint 저장이 test 평가를 뜻하지는 않는다.

PCA와 Ridge는 먼저 별도로 적합한 뒤 고정한다. DDPM loss가 갱신하는 것은 context encoder·time embedding·head의 신경망 가중치다. 전체 pipeline이 Ridge/PCA까지 공동으로 end-to-end 학습되는 것은 아니다. 별도의 신경망 사전학습은 없다.

## 교수님 질문에 대한 짧은 답변

| 질문 | 답변 |
| --- | --- |
| 왜 RMA 후 표준화? | RMA는 상위 마이크로어레이 전처리, 학습 표준화는 probe별 모델 입력 척도 조정이다. |
| 왜 baseline 두 개? | 변화0은 최소 처리 반응 기준, 조건 평균은 추가 발현·구조 없는 공통 반응 기준이다. |
| 왜 Ridge? | 공통 목표 조건 반응과 추가 입력의 잔차 기여를 분리하려는 설계다. |
| 왜64/32/128? | 압축과 모델 규모를 제한하려는 초기값이며 일부 width 검증과 현재 조합 탐색으로 민감도를 확인 중이다. 전역 최적값은 아니다. |
| cGAN보다 좋으면 AIVIVE보다 좋은가? | 아니다. 현재 틀의 기본 cGAN과 비교했고 AIVIVE 공식 재현은 하지 않았다. |
| 구조가 도움이 됐나? | 고정 분할의 작은 평균 개선 신호다. 무작위 fingerprint 특이성·scaffold 외삽·독립 확증은 아니다. |
