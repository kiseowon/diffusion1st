# 입력 PCA만 유지하고 전체 probe를 직접 출력하기

`meeting_time_structure_pilot.py --output-space full`은 입력 시간 슬롯과 입력 PCA를 유지하면서 **출력 PCA를 제거**한다. 기본값 `pca`와 기존 결과는 보존한다. 새 출력 폴더에서 실행해야 하며, 옛 결과와 혼합해 재개할 수 없다.

## 정답 보존

정답은 준비 데이터의 모든 3,453개 probe 반응이다. 해당 학습 집합에서 probe별 평균 `mu`를 구하고, 중심화한 전체 target 값에 대해 공통 RMS `s`를 구한다.

```text
학습 target z = (실제 probe 반응 y − mu) / s
모델 출력: 각 probe에 해당하는 z_hat 3,453개
예측 반응 y_hat = s × z_hat + mu
```

차원 축소·출력 projection·target clipping·probe 제거는 없다. 거의 0인 공통 스케일은 1로 대체한다. 공통 스케일이므로 probe별 상대 가중치를 바꾸지 않는다. 변환은 수학적으로 가역이며 부동소수점 반올림 오차는 남는다. `target_representation.json`에 실제 출력 수와 float64 학습 target의 변환-역변환 최대오차를 저장한다. 모델 입력/학습 tensor는 기존처럼 float32이다.

출력 PCA가 강제하는 고정 부분공간의 한계를 없애는 것이지 모델의 표현력·예측 정확도를 보장하는 것은 아니다. 기존 작은 hidden64를 유지하므로 learned bottleneck의 용량 한계는 여전히 있다. 모든 probe가 학습 손실에 포함된다.

## 유지하는 입력과 비교

화합물별 시간·용량 슬롯, 관측mask, 학습 source MinMax→입력 PCA 최대32→score 표준화, source/target dose/time/unit/level, 구조 없음/Morgan/bounded 2D Descriptor를 유지한다. Descriptor는 별도 signed-log·학습 열 선별·bounded MinMax 정책이고 출력 반응에는 적용하지 않는다. RMA를 다시 수행하지 않는다.

MLP와 cGAN은 전체 probe 벡터를 출력한다. cGAN의 복원 손실도 전체 probe MAE이다. DDPM은 전체 probe 공간의 noisy 반응에서 x0를 예측한다. 복원 MAE/MSE는 공통 스케일을 제외하면 원래 probe 공간의 동일 가중 MAE/MSE와 동등하다. cGAN의 BCE 항은 별도이므로 전체 생성 손실까지 단순히 동일하다는 뜻은 아니다.

context256/head64, Adam lr1e-4, batch128, DDPM20step/sample4, gradient clip1, 내부 검증 및 선택 후 외부 재학습 정책은 유지한다. cGAN D는 SGD lr1e-4/momentum0.9이며 cycle은 없다. 최대epoch100을 명시해 실행할 수 있다. 원래 AIVIVE의 대형 생성기나 손실 전체를 재현한 것은 아니다.

최초 검사는 같은 분할 seed42, fold0, model seed42에서 세 구조 입력 × 세 모델=9조건으로 수행한다. 5fold 전체 반복 결과가 아니다. full 출력 head와 noisy 입력이 커져 parameter 수와 비용이 증가한다. 소프트 시간 제한은 현재 job을 강제 중단하지 않으므로 완료 시간을 보장하지 않는다. 완료된 결과는 중간 저장하고 같은 snapshot·입력·설정·출력 폴더로 재개한다.

## 출력·해석

기존처럼 예측npz, 모델pt, 전처리joblib, target/control audit, 검증 이력, 입력/소스 hash를 저장한다. results.csv와 classification.csv는 역변환한 전체 probe 공간에서 계산한다. test의 실제 target은 학습이나 전처리 적합에 사용하지 않는다. 기준선은 같은 target에서 zero response와 훈련 시간×등급 평균이다.

이전 PCA 출력과 비교할 때는 같은 fold/seed/평가 target을 사용해야 한다. 5fold 전체 평균과 한 fold의 값을 직접 비교하지 않는다. 출력 차원과 출력층 parameter 수, 손실이 계산되는 공간이 함께 바뀌므로 차이를 압축 손실 하나의 효과로 단정하지 않는다. `--reference-run`으로 옛 PCA 출력 점수를 자동 병합하는 것은 거부한다.

반복 실행기는 `repeat_time_structure_pilot.py --output-space full`을 지원한다. 새 폴더에서 사용하며, 이 경우 전체9조건을 원하면 `--arms basic morgan descriptor`를 지정한다. 우선 단일holdout의 비용과 학습 곡선을 보고 확대한다.

```bash
python scripts/meeting_time_structure_pilot.py \
 --prepared /home/seowon1211/data1/conditioned_expression/prepared_controls \
 --structure /home/seowon1211/data1/conditioned_expression/morgan_matched/morgan_binary.csv \
 --smiles /home/seowon1211/data1/compound_pubchem_smiles.csv \
 --out /home/seowon1211/data1/conditioned_expression/NEW_FULL_OUTPUT_RUN \
 --output-space full --epochs 100
```

기본은 계획만 출력한다. 실제 학습에는 `--execute`를 붙인다. cloud에서는 전체 차원 유지·target 역변환·held-out 정답 미사용·9조건 저장/재개를 합성 데이터로 검사했다. 연구 자료의 학습 성능이나 재현을 확인한 것은 아니다.
