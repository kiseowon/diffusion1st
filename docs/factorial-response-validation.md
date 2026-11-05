# 직접 반응 예측: 네 입력 × 세 모델

`scripts/factorial_response_validation.py`는 새 탐색 실험용이다. 기존 결과와 residual 기본 모드는 보존한다. 실제 데이터 학습은 이 클라우드에서 실행하지 않았다.

| arm | 공통 입력 | Morgan | KEGG 고정 평균 |
|---|---|---|---|
| basic | in vitro 반응 + 출발·목표 실험 조건 | 0 | 0 |
| structure | 동일 | 실제 | 0 |
| pathway | 동일 | 0 | 실제 |
| both | 동일 | 실제 | 실제 |

각 arm에서 MLP/cGAN/x0-DDPM을 학습한다. 'basic'은 입력 없음이 아닌 추가 구조·pathway 없음이다. 제거한 특징 자리는 0으로 유지해 네 arm의 네트워크 입력 크기·명목 parameter 수를 맞춘다. 활성 특징과 유효 자유도가 동일하다는 뜻은 아니다. 모든 encoder는 최종 모델 손실로 학습되며 별도 사전학습은 없다.

## 입력과 전처리

필수 입력은 prepared_controls의 vitro/vivo_expression.npy, vitro/vivo_conditions.csv, pairs.csv.gz, probes.csv, Morgan binary CSV, KEGG projection.npy와 provenance.json이다. 다운로드나 매핑 재생성은 하지 않는다. provenance의 probe_order가 probes.csv와 정확히 같아야 한다. projection은 유한한 비음수이며 각 열의 합이 1이어야 한다. hash와 코드 버전은 각 arm/run.json에 보존한다.

화합물·시간·단위가 같은 실제 대조군을 빼 source와 target 반응을 만든다. 학습 고유 source로 probe별 StandardScaler를 적합한다. 기존의 target별 source 평균 정책을 유지한다. source 반응 PCA는 최대 64차원이며 학습 집합으로만 적합하고 작은 singular value 축을 제거한 뒤 score를 표준화한다. 조건은 log1p 용량·시간 표준화와 단위·등급 one-hot이다. 단위의 물리적 환산이나 이름 자동 수정은 하지 않는다. Morgan은 이진값과 presence이며 미매칭은 0+presence 0이다.

Pathway는 표준화한 source probe 반응에 저장된 projection 가중치를 곱한 고정 평균이다. 유전자당 총 기여도를 맞춘 원본 projection을 그대로 사용한다. 각 target에 연결된 source들의 pathway score를 평균하고 학습 target으로 적합한 StandardScaler를 적용한다. 입력 PCA와 병렬로 모델에 제공하므로 PCA 압축에서 잃은 정보도 담을 수 있다. 예전 sparse encoder, 학습 가능한 pathway 연결 가중치, 그래프 메시지 전달과는 다르다. 외부 지식은 KEGG membership이고 pathway 값 자체는 입력 발현에서 유도된다.

정답은 vivo 처리군−대조군 반응이다. 학습 정답에만 PCA 최대 32차원을 적합하고 잠재 score의 공통 RMS로 나눈다. probe별 target 분산을 각각 나누지 않는다. 예측 score를 역변환하면 학습 target 평균이 다시 더해진다. 이것은 일반적인 PCA 복원이며 Ridge 보정은 아니다. 저차원 출력이므로 probe 전체 직접 생성과 구분한다.

## 모델과 선택

세 모델 모두 context 64, hidden 128의 기존 작은 모델을 사용한다. MLP는 잠재 반응 MSE, cGAN은 조건부 BCE만 사용하며 generator/discriminator를 각각 학습한다(재구성 손실 없음). DDPM은 cosine schedule 30단계의 x0 예측 MSE이며 예측은 16개 생성 평균이다. 생성모델의 개별 동물 분포 품질을 평가하는 설계가 아니다.

AdamW lr=0.0003, weight_decay=0.01, batch=64, clip=1, 최대 100 epoch. 내부 검증 epoch 1/25/50/75/100에서 원래 probe 공간 response RMSE로 선택한다. 이후 외부 학습 전체에 전처리와 모델을 다시 적합한다. Ridge는 학습하지 않으며 보정 강도 선택도 없다(항상 1). 기존 잔차 모델의 width 검증에서 hidden64가 더 낮았다는 사실을 이 직접 예측 모델의 최적성 증거로 옮기지 않는다. 위 값들은 고정된 초기 비교 설정이며 각 모델의 최적값이 아니다.

## 분할·출력·해석

화합물 5-fold, split seed42, model seed42/43/44를 모든 arm에 공유한다. 외부 학습 안에서 화합물 20%를 내부 검증으로 둔다. 기본 180조건, 최대 360 neural fit이며 합성 데이터 검사와 실제 재학습 재현은 구분한다. 기존에 같은 화합물의 결과를 관찰했으므로 독립 확증 시험이 아니다.

각 arm에 예측·checkpoint·전처리·분할·검증 이력을 보존한다. `factorial_summary.csv`는 전체 target×probe 오차를 합친 RMSE이며 seed별 MSE를 평균한 것으로 예측 ensemble이 아니다. `factorial_paired_differences.csv`에 structure−basic, pathway−basic, both−basic, both−structure, both−pathway를 기록한다(음수가 개선). RMSE 척도의 interaction은 기술 통계로만 읽고 검정으로 해석하지 않는다.

기준선은 실제 학습 화합물로부터 화합물별 평균 후 화합물 간 평균한 시간×목표 등급 반응과 zero response다. 테스트 대조군은 정답 반응 계산에만 사용한다. 기준선은 반응만 예측하므로 absolute RMSE를 만들지 않는다. 추가 평가는 기존 evaluate_repeated_predictions.py로 각 arm을 지정해 MAE/R2/상관/F1 등을 계산할 수 있다.

새 실행은 아래 명령에 `--execute`를 추가한다. 기본은 입력을 읽지 않고 출력도 만들지 않는 계획 모드다. 동일 코드·입력·설정일 때만 중단된 실행을 재개한다.

```bash
python scripts/factorial_response_validation.py \
  --prepared /home/seowon1211/data1/conditioned_expression/prepared_controls \
  --structure /home/seowon1211/data1/conditioned_expression/morgan_matched/morgan_binary.csv \
  --kegg /home/seowon1211/data1/conditioned_expression/kegg_rat_20261102_034010_265aed \
  --out /home/seowon1211/data1/conditioned_expression/NEW_FACTORIAL_RUN
```

이 비교는 추가 특징의 실용적 효과를 보는 초기 ablation이다. 무작위 membership·셔플 구조 대조, 하이퍼파라미터 최적화, scaffold 분할은 포함하지 않는다. 특히 pathway 개선만으로 생물학적 연결의 고유 효과나 인과성은 주장할 수 없다. 작은 데이터에서 고정 32차원 출력·BCE cGAN 등 설계 제한과 반복 관찰한 테스트의 탐색 성격을 함께 보고한다.
