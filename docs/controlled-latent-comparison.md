# 같은 잠재 공간에서 MLP와 x₀-DDPM 잔차 보정 비교

## 질문과 기록 범위

동일한 입력·반응 압축·기본 예측 아래에서, 잠재 반응을 직접 예측하는 MLP보다 조건부 Diffusion으로 생성·평균하는 방식이 더 낮은 오차를 보이는가?

설명은 채팅에서 제공한 실행 셀을 근거로 하고, 결과는 사용자가 제공한 서버 로그를 근거로 합니다. 서버에서 실제 실행한 파일과 채팅 셀의 checksum 일치는 아직 확인하지 못했습니다. 이 환경에서 학습을 실행하거나 원자료를 독립 검증한 것은 아닙니다.

## 데이터와 분할

- 연구 대상: Rat liver, 고정된 3,453개 probe. 기존 연구에서 사용한 140개 화합물.
- 정답: vivo 처리군 발현에서 같은 compound·unit·time_h의 실제 대조군 발현을 뺀 조건 평균 반응.
- 입력 발현도 vitro 처리군−같은 compound·unit·time_h 대조군의 변화량입니다.
- source/target pair 연결과 저장 probe 순서는 기존 prepared_controls를 사용합니다. 같은 vivo target에 연결된 여러 source의 **입력 특징을 먼저 평균**합니다. 과거 source별 예측 후 평균한 실험과 다릅니다.
- split seed42의 화합물 단위 내부 학습89 / 검증23 / 외부 테스트28. 설정 선택 후 112개 화합물로 다시 적합합니다. 모델 seed는42 하나입니다.
- 기존 decoder 셀의 전처리·분할·기준선 객체를 재사용하는 continuation 셀이었습니다. 새 커널에서 단독 실행할 수 있는 완결 스크립트는 아닙니다.

실제 분할 CSV·전처리 객체 원본은 아직 제출되지 않아 위 내용은 제공 코드 기준 설계 기록입니다. 원자료의 RMA/log2 처리 여부는 값 범위만으로 확정하지 않습니다.

## 공통 입력과 전처리

1. 학습 화합물의 고유 source 조건으로 vitro 변화량의 probe별 StandardScaler를 적합합니다.
2. target별로 연결된 source의 표준화 발현을 평균하고, 학습 target 행으로 입력 PCA를 적합합니다. 최대64축이며 singular value가 최대값의 1e-8 이하인 축은 제외합니다.
3. 살아남은 입력 PCA score는 학습 행으로 StandardScaler를 적합합니다.
4. source와 target 각각 dose·time·unit·level을 조건으로 사용합니다. dose에는 log1p, 수치 열에는 학습 pair 기반 표준화, 범주에는 unknown을 무시하는 one-hot을 적용합니다. 조건 특징도 target별 pair 평균을 사용합니다. 단위의 물리적 환산은 추가하지 않습니다.
5. Morgan binary와 presence flag는 표준화하지 않고 입력에 연결합니다. 누락 구조는0과 presence=0으로 처리하는 기존 로더를 따릅니다. 구조 로더의 이름 매칭 검사가 물질 동일성을 보증하지는 않습니다.

입력 PCA/scaler는 내부 학습 단계와 최종 refit 단계에 각각 따로 적합합니다. Morgan 특징은 실제 실행 시 상위 셀의 data.structure를 재사용했습니다. 그 셀은 morgan_matched/morgan_binary.csv를 읽도록 제공됐지만, 서버 객체·입력 hash 원본은 아직 확인하지 못했습니다.

## 공통 목표 조건 Ridge

목표 vivo dose(log1p)·time을 표준화하고 unit·level을 one-hot으로 변환합니다. **in vitro 발현이나 Morgan은 이 기본 Ridge의 입력이 아닙니다.**

후보 α=10,100,1000 중 내부 검증의 반응 RMSE로 선택합니다. 상위 decoder 실행 로그에서는100이 선택됐습니다. 동일하게 선택한 α로 외부 학습112개 화합물에 전처리·Ridge를 다시 적합합니다.

시간×등급 평균 기준선과는 다른 기준선입니다. 평균 기준선은 별도로 보고된 수치이며 이번 실행에서 다시 계산한 것이 아닙니다.

## 출력 잔차 PCA

훈련 반응에서 해당 훈련 자료로 적합한 Ridge 예측을 뺍니다.

`residual = actual_response - baseline_ridge_prediction`

이 실험은 **훈련 내 Ridge 잔차**를 사용합니다. OOF 잔차 모델이 아닙니다. 이것 자체가 외부 시험 누수는 아니지만 훈련과 새 화합물의 잔차 분포가 달라질 수 있습니다.

훈련 잔차로 최대32차원 PCA를 적합하고, 거의0인 축을 singular value의 상대 기준1e-8로 제외합니다. PCA는 평균을 제거합니다. 살아남은 잠재 score 전체에 대해 훈련 공통 RMS 한 개를 계산해 나눕니다. 축별 whitening은 하지 않습니다.

생성·예측한 score에 공통 RMS를 곱하고 PCA 성분으로 probe 공간에 복원한 뒤 PCA 평균 잔차를 더합니다. 보정 강도 λ는 **평균 잔차를 포함한 복원 보정 전체**에 곱합니다.

`final_response = baseline_ridge_prediction + λ * restored_residual`

내부와 최종 학습에서는 PCA·공통 RMS를 각각 다시 적합합니다. 코드의32·64는 최대값이며 실제 유효 축 수는 서버 저장 객체로 확인해야 합니다.

## 두 예측기의 차이

공통 형태는 `입력 → Linear64+SiLU → [잠재입력, time embedding16과 결합] → Linear128+SiLU → 출력 잠재값`입니다.

- MLP: 잠재입력0·time index0을 고정하고 정답 잠재값의 MSE를 최소화합니다.
- x₀-DDPM: 학습 시 t를 균등 추출하고, 정답 잠재값에 schedule에 따른 Gaussian noise를 섞습니다. 깨끗한 잠재값 x₀의 MSE를 최소화합니다. epsilon 예측기가 아닙니다.
- 생성: cosine beta schedule,30단계의 DDPM posterior 역과정.16회 생성한 잠재값을 평균합니다. 마지막 단계에는 추가 noise가 없습니다. 별도 clipping을 사용하지 않습니다.

명목 network 형태는 같지만 MLP는 noisy/time 입력이 고정돼 유효 자유도는 같지 않습니다. DDPM에는 stochastic 생성과16회 평균의 추론 비용이 추가됩니다. 완전히 같은 연산 예산의 비교나 모든 모델의 최적 튜닝은 아닙니다.

KEGG·무작위 연결·graph·FiLM·AE·VAE는 이번 예측기에 사용하지 않습니다. 상위 셀 이름에 pathway가 있어도 이번 입력과 출력 예측기는 pathway 정보를 사용하지 않습니다.

## 선택과 재학습

- AdamW lr0.0003, weight_decay0.01, batch64, gradient norm clip1.
- 최대100epoch.1,25,50,75,100에서 검증합니다. 미평가 epoch가 있으므로 전 epoch 중 최적이라고 주장하지 않습니다.
- 각 평가 시 λ=0,0.1,0.25,0.5,1을 비교합니다. 처음부터 Ridge만 사용하는 epoch0·λ0을 후보에 둡니다.
- 검증 생성 seed20042, 최종 테스트 생성 seed30042. 모델 초기화·학습 seed42. CPU,4threads.
- epoch·λ를 내부 검증 response RMSE로 선택한 뒤 모델을 초기화하고 외부 학습112개에서 해당 epoch만큼 재학습합니다. 최종 단계에 epoch checkpoint의 학습 가중치를 그대로 가져가는 방식은 아닙니다.
- 각 모델의 외부 테스트는 선택 완료 후 평가합니다. 다만 연구 전체에서는 같은 테스트를 보며 설계를 발전시켰으므로 확증적 독립 시험은 아닙니다.

32/64차원·학습률·width·100epoch·30step·16sample은 제한된 탐색 설정으로, 최적성 검증이 없습니다. MLP가 상한100epoch를 선택했다는 사실만으로 충분한 최적화나 학습 부족을 단정하지 않습니다.

## 보고된 결과

| 방법 | 테스트 RMSE | 선택 epoch | λ | 검증 RMSE |
| --- | ---: | ---: | ---: | ---: |
| 목표 조건 Ridge | 0.3128388253287149 | — | — | — |
| Ridge+잠재 MLP | 0.3118587617478962 | 100 | 0.1 | 0.24398470772839295 |
| Ridge+잠재 x₀-DDPM | 0.3106236005192594 | 75 | 0.25 | 0.2432111596190763 |

RMSE는 고유 treated target별로 평가한 전체 target×probe 제곱오차 평균의 제곱근입니다. compound macro-average가 아니며 target·probe별 중요도 차등 가중은 없습니다. 절대 발현 RMSE·개별 동물의 생성 분포 품질은 평가하지 않습니다.

DDPM−MLP = −0.0012351612286368, DDPM−Ridge = −0.0022152248094555입니다. 이는 보고 수치의 산술 차이이며 독립 지표 재계산은 아닙니다. 이전 시간×등급 평균 기준선0.3103845757483401보다 DDPM 오차는 약0.000239 높습니다. 이 기준선 비교는 동일 분할이라는 상위 실험 기록에 따른 맥락이며 서버 prediction/target ID로 재대조하지 않았습니다.

## 해석 범위와 남은 자료

이 분할·초기화에서 DDPM 보정이 잠재 MLP보다 낮은 오차를 보였다는 신호입니다. 구조 on/off 대조가 없어 구조 효과를 주장할 수 없고, pathway 비교도 아닙니다. 단일seed,단일분할,선택된 검증집합,누적된 테스트 관찰 때문에 유의성·일반적 우월성·불확실성 calibration을 주장하지 않습니다. 이전5-fold PCA 잔차 Diffusion 결과와 합치거나 직접 순위를 매기지 않습니다.

서버 실행 폴더: `/home/seowon1211/data1/conditioned_expression/controlled_latent_mlp_ddpm_20261104_044442_34d1be`.

아직 필요한 원본은 executed_cell.py, config.json, parent.json, parent 분할·입력 hash 기록, baseline NPZ, 모델 소속이 확인된 각 prediction NPZ·checkpoint, 전처리 joblib입니다. 두 모델의 validation_history.csv·selection.json·complete.json·condition_metrics.csv는 추가 첨부로 확보했습니다. 특히 상위 커널의 의존 객체가 있어 두 실행 셀과 분할·전처리를 함께 보존해야 합니다.

현재 GitHub에는 설계와 전사된 결과 및 일부 첨부 산출물을 보존했습니다. 원자료·실제 실행 코드 snapshot·모델별 산출물 전체의 백업 완료, 학습 재현 완료, 서버 코드 hash 일치는 주장하지 않습니다. 이 기록을 만들면서 추가 실험은 실행하지 않았습니다.

## 추가 첨부 산출물의 검사

MLP와 DDPM 각각 selection.json·complete.json·validation_history.csv·condition_metrics.csv를 원본 바이트 그대로 보존했습니다. 파일명 (1) 자체로 모델을 결정하지 않고 complete의 model 필드, 선택 epoch·강도·검증 점수, 검증 이력을 함께 대조했습니다. 두 조건별 표의 336개 고유 target,28개 화합물 및 condition_id·compound·time_h·level 순서가 일치합니다. 조건별 MSE 평균의 제곱근은 각각0.3118587617478962와0.3106236005192594로 완료 기록과 일치합니다.

첨부된 predictions.npz 하나는336×3453, 유한한 값이며 target ID 순서는 두 모델의 지표 표와 같습니다. ID가 같다는 것으로 모델 소속을 결정할 수는 없습니다. model.pt 하나도 모델 소속 미확인 상태로 별도 보존했고 ZIP CRC만 검사했습니다. 자동 승인 검토의 최초 전송 거부 이후 사용자가 외부 업로드를 명시적으로 승인하여 두 파일 원본도 unassigned/에 보존했습니다. torch.load나 pickle 실행은 하지 않았습니다. 두 파일을 특정 모델의 확정 산출물이라고 표시하지 않습니다.

artifact_manifest.json은 첨부 파일별 원래 이름·업로드 ID·보존 위치·SHA-256을 기록합니다. artifact_checks.json은 검사 결과입니다. checksum은 받은 파일의 보존 근거이며 서버 원본과 독립적으로 대조한 근거는 아닙니다. 원자료 정답이 없어 예측으로부터 RMSE를 직접 재계산하지 못했고, 실행 셀·입력·분할·전처리 원본은 여전히 필요합니다.
