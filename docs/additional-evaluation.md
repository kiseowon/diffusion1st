# 저장 예측의 추가 평가: 회귀·방향·3분류 F1

`scripts/evaluate_repeated_predictions.py`는 완료된 repeated_latent 실행 폴더의 예측을 읽는 **평가 전용 코드**입니다. 모델·joblib을 로드하지 않고, 학습·예측 재생성을 하지 않습니다. 원래 prepared_controls의 정답에서 같은 compound·unit·time_h 대조군을 빼 실제 반응을 복원합니다.

학습 run.json의 원자료 hash를 먼저 재검사하고, 모델별 complete.json에 기록된 산출물 hash를 확인합니다. 각 fold는 지정된 시험 화합물의 target을 중복 없이 한 번씩 예측해야 합니다. 계산한 조건 MSE가 저장된 condition_metrics.csv와 일치해야 추가 평가를 진행합니다. 원자료가 바뀌었거나 유효 완료 기록이 없으면 중단합니다.

## 지표 정의

| 지표 | 계산·의미 |
| --- | --- |
| RMSE / MAE | 전체 OOF target×probe의 제곱오차/절대오차 기반. 낮을수록 좋음 |
| Ridge/시간×등급 평균 대비 개선율 | 100×(1−모델RMSE/기준선RMSE). 음수이면 악화. 예측 정확도%가 아님 |
| R² | probe마다 전체 OOF target의 실제 평균을 기준으로 SST 계산 후 1−전체SSE/전체SST. probe 분산 가중 multioutput R²이며 음수가 가능. 전체SST0이면 미정의 |
| Pearson target mean | target마다3453probe의 실제·예측 반응 상관을 계산해 산술평균. 전체 flatten 상관이나 probe별 시간·화합물 상관과 다름 |
| Spearman target mean | 같은 target별 probe 순위 상관의 산술평균 |
| 3분류 정확도 | 실제·예측 각각 동일 임계값으로 down/small/up을 나눠 맞힌 비율 |
| Macro F1 | down/small/up 3개 F1의 동일 가중 평균. support와 예측이 없는 class의 F1은0으로 고정 |
| Weighted F1 | 실제 class support로 가중한 F1. small 다수 class에 영향받을 수 있음 |
| Active sign accuracy | 실제 절댓값≥τ인 원소에서 예측 부호가 같은 비율. 예측 절댓값이τ보다 작아도 부호만 같으면 맞음 |
| Class별 precision/recall/F1/support | 특정 방향만 놓치거나 small만 예측하는지 확인 |

표준화된 잠재 공간이 아니라 **역변환된 원래 반응 척도**에서 평가합니다. 이후 사용자 서버의 배포 checksum 일치로 [AIVIVE RMA 입력 출처](expression-source.md)가 연결됐습니다. 기본τ=0.5와1.0은 결과를 보기 전에 지정한 탐색적 진폭 threshold이며 통계적으로 유의한 차등발현,2배 변화,검증된 생물학적 중요도 기준이 아닙니다. 유리한 threshold를 사후 선택해 주결과로 바꾸지 않습니다.

분류 규칙은 `y≥τ: up`, `y≤−τ: down`, 나머지 `small`입니다. 모든3453probe를 포함합니다. 변화가 작은 원소가 많으면 전체 accuracy가 높아도 반응 방향을 못 맞힐 수 있으므로 Macro F1·class별 support·active sign accuracy를 함께 봅니다. 실제 변화가 큰 원소가 없으면 active sign accuracy는 빈 값입니다.

상수인 실제/예측 벡터는 상관계수가 정의되지 않아 target 평균에서 제외하고 valid target 수를 함께 기록합니다. 모두 미정의이면 평균도 빈 값입니다. 상관은 반응 크기 정확도를 보장하지 않으므로 RMSE·MAE를 대체하지 않습니다.

## 실행

```bash
python scripts/evaluate_repeated_predictions.py \
  --run /완료된/repeated_latent_폴더 \
  --out /새로운/평가결과_폴더 \
  --thresholds 0.5 1.0
```

numpy,pandas,scipy만 사용하며 torch 및 sklearn 모델 역직렬화는 필요하지 않습니다. 학습 폴더와 원자료는 수정하지 않습니다. 출력은 별도 새 폴더만 허용합니다. 기본은 완료된 실행의 모든 model seed와 기준선을 평가합니다. 학습 epochs를 다시 실행하는 코드가 아닙니다.

## 출력과 집계

- regression_by_unit.csv: split seed·model seed별 전체OOF 회귀 지표. baseline은 model_seed=−1로 split당 한 번만 평가합니다.
- classification_by_unit.csv: 같은 단위·threshold별 분류/방향 지표. `_pct` 열은0~100표시입니다.
- regression_summary.csv,classification_summary.csv: unit별 지표의 mean/std. 3개 seed를 독립적인 시험 화합물 집합으로 간주하지 않습니다. baseline unit이 하나면 std는 빈 값입니다.
- confusion_matrices.csv: actual class가 행, predicted class가 열인9개 count. probe×target의 count이지 독립 표본 수가 아닙니다.
- target_metrics.csv: target별 RMSE·MAE·상관. 전체 classification 성능은 target/probe 균등 집계이며 compound macro 지표가 아닙니다.
- evaluation.json: 평가 source hash·입력 확인 hash·산출물 확인 hash·threshold·환경.
- completed.json: 모든 unit 평가 완료.

summary의 RMSE는 **unit별 RMSE의 평균**입니다. 원래 반복 실행기 summary.csv는 seed/repeat MSE를 먼저 평균한 뒤 제곱근을 취하므로 미세하게 다를 수 있습니다. 어느 것도 model prediction을 ensemble한 성능은 아닙니다. paired 비교나 통계적 우월성 판단에는 unit별 기록과 화합물 의존성을 고려해야 합니다.

평가 코드는 완벽 예측·상수 상관·sklearn의 fixed3class F1과 일치·축소 합성 저장 예측의 전체 평가·재학습 금지·입력 변조 거부를 검사했습니다. 이는 연구 원자료의 새로운 결과 검증을 완료했다는 뜻이 아닙니다. 추가 지표는 주RMSE를 보고 난 뒤 요청된 **사후 탐색 평가**임을 유지합니다.

새 run의 `families`에 cgan이 있으면 같은 방식으로 추가 평가합니다. 해당 필드가 없는 이전 run은 mlp/ddpm으로 해석합니다.
