# AIVIVE를 참고한 PCA 없는 반응 예측

실행기: `scripts/aivive_inspired_validation.py`. 기존 결과·실행기는 보존하며 새 폴더에만 출력한다. 기본은 계획 모드이며 `--execute`에서만 학습한다. 이 환경에서는 합성 데이터 검사만 수행했다. 실제 데이터의 성능·수렴은 미확인이다.

## 참고한 출처와 적용 범위

확인한 출처는 AIVIVE 공개 저장소 commit `c0a4bd3e5b31b30031eb4bd103bf92aaef552275`의 [training/vitro_vivo_GAN.py](https://github.com/CHANDMX20/AIVIVE/blob/c0a4bd3e5b31b30031eb4bd103bf92aaef552275/training/vitro_vivo_GAN.py)이다. 공개 코드에서 확인한 설정이며 논문 결과 생성 버전과 동일하다고 독립 검증하지 않았다.

| 항목 | 공개 코드 | 이번 적용 |
|---|---|---|
| 발현 변환 | vitro/vivo 별 학습 MinMaxScaler, PCA 없음 | source/target **반응**에 각각 학습 MinMaxScaler, PCA 없음 |
| Generator 최적화 | Adam lr 0.00001 | 세 예측 모델에 Adam lr 0.00001, weight decay 0 |
| 판별기 최적화 | SGD lr 0.0001, momentum 0.9 | 동일 |
| composite 손실 | 직접 MAE : BCE : cycle MAE = 10:2:5 | cGAN 10 MAE + 2 BCE, cycle 없음 |
| 은닉층 | Generator 8192→7168→7168→4096→4096, D 256→64 | 소형 context256/head64. 숫자는 판별기 폭을 참고한 적응 설계이며 원래 Generator 구조가 아님 |
| 출력 | sigmoid | 선형. 학습 범위를 넘는 반응을 허용 |
| dropout | G 0.8/0.8/0.8/0.4, D 0.5/0.5 | 없음. 이 강한 규제를 작은 모델로 이전하지 않음 |
| batch / epoch | 1 / 10000 | 64 / 최대100. 계산 예산을 제한한 초기값 |
| 데이터 단위 | 발현 profile과 관련 label | 조건 평균 treated−control 반응 및 출발·목표 조건 |

공개 코드의 모든 설정을 복사한 재현 실험이 아니다. 학습률과 손실 가중치의 출처가 있다는 것과 이 데이터에서 최적이라는 것은 다르다. 100 epoch와 작은 학습률 조합에서 학습 부족이 생길 수 있으며, epoch100이 선택되거나 검증 곡선이 계속 개선되면 수렴 제한을 보고해야 한다. 결과를 보고 곧바로 테스트에 맞춰 재조정하지 않는다.

## 비교와 입력

구조 없음/있음 × MLP/cGAN/x0-DDPM의 6조건이다. 기본은 화합물5fold, split seed42, model seed42/43/44: 총90조건, 최대180 neural fit. 기존 동일 분할의 테스트를 관찰했으므로 탐색 실험이다. 모델 seed 반복은 새로운 화합물 분할 반복이 아니다.

prepared_controls에서 compound/unit/time이 같은 대조군을 빼 source/target 반응을 만든다. 학습 source 조건만으로 probe별 MinMaxScaler를 적합한다. 같은 target에 연결된 source 입력과 source 조건들을 평균한다. dose log1p+표준화, 시간 표준화, 단위·등급 onehot은 기존 정책을 유지한다. 구조가 있으면 Morgan2048+presence를 원래 이진값으로 결합하고 없으면 해당 자리를 모두 0으로 유지한다. 누락 구조·단위 문제·이름 문제를 자동으로 고치지 않는다.

정답의 학습 target만으로 별도 probe별 MinMaxScaler를 적합하고 모든 probe를 출력한다. validation/test truth로 scaler를 적합하지 않는다. clip하지 않으므로 held-out 값은 [0,1] 밖일 수도 있다. 출력은 선형이므로 [0,1]로 강제하지 않는다. inverse_transform 후 원래 반응 척도에서 평가한다. probe별 범위가 달라 scaled loss는 원래 probe 공간의 동일 가중 MAE/MSE와 다르다. 상수 probe는 sklearn의 기본 MinMax 처리를 그대로 사용하며 제거하지 않는다. RMA나 log 변환을 재실행하지 않는다.

- MLP: context256→head64→전체 probe, LeakyReLU0.2, scaled MAE.
- cGAN: context256 + 정규 noise16→head64→전체 probe. 별도 D는 context256과 probe를 결합→64→logit. batch당 D1회/G1회; G 손실10MAE+2BCE. cycle·양방향 generator 없음. 16회 생성 평균.
- DDPM: context256 + noisy 전체 probe + timestep embedding16→head64→전체 probe. x0-MSE, cosine30step, 16회 생성 평균. PCA/Ridge/잔차 보정 없음.

세 모델은 공유한 입력·분할에서 비교하지만 학습 목적함수는 모델별로 다르다. 계열 전체의 최적 성능이나 AIVIVE 원본과의 우열을 의미하지 않는다.

## 선택·출력

내부 학습/검증 화합물로 분리하여 epoch1/25/50/75/100에 역변환한 response RMSE로 선택한다. 선택 후 외부 학습 전체에 scaler와 모델을 새로 적합한다. 출력은 각 arm/basic 또는 structure에 전처리·checkpoint·예측·분할·검증 이력·입력/코드 hash·패키지 버전을 저장한다. 각 arm은 원래 반복 실행기의 정확한 설정·입력·소스 계약으로 재개한다.

평가 기준선은 zero response와 화합물별 평균 후 화합물 간 평균하는 훈련 시간×등급 반응이다. 둘 다 실제 반응을 예측하므로 absolute expression RMSE를 만들지 않는다. `structure_summary.csv`는 target×probe MSE를 seed에 걸쳐 평균한 pooled RMSE이며 예측 ensemble이 아니다. 기존 evaluate_repeated_predictions.py에 각 arm을 전달하면 MAE/상관/F1도 재학습 없이 계산한다.

MinMax, 손실, 모델 폭 등 여러 요소를 동시에 바꾸므로 기존 PCA 실험과의 차이를 **PCA 제거 효과 하나**로 해석할 수 없다. 이 효과만 분리하려면 나머지 설정을 고정한 별도 ablation이 필요하다.

```bash
python scripts/aivive_inspired_validation.py \
 --prepared /home/seowon1211/data1/conditioned_expression/prepared_controls \
 --structure /home/seowon1211/data1/conditioned_expression/morgan_matched/morgan_binary.csv \
 --out /home/seowon1211/data1/conditioned_expression/NEW_AIVIVE_INSPIRED_RUN
```

위 명령은 계획만 출력한다. 학습하려면 `--execute`를 추가한다. CPU threads4가 기본이며 사용자 GPU에서의 실행 검증은 하지 않았다.
