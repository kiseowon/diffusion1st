# DDPM 설정 이유와 내부 검증 민감도

## 설계 의도와 최적성은 다르다

현재 설정은 작은 데이터에서 반응 압축과 작은 예측기를 사용하는 토이 실험의 고정 초기값입니다. 특정 논문이 이 숫자를 최적이라고 권장해서 택한 것이 아닙니다. 과거 검증에서 epoch·보정 강도를 선택했다는 사실과, PCA차원·network폭·학습률까지 최적화했다는 주장을 구분합니다.

| 설정 | 설계 목적 | 기존 검증 범위 |
| --- | --- | --- |
| 입력 PCA 최대64 | probe 발현을 작은 입력 표현으로 요약 | 64가 최적인지 미검증 |
| 출력 잔차 PCA 최대32 | 생성·예측할 반응 차원과 계산량 제한 | 반응 정보 손실 및 최적 차원 미검증 |
| context64 | 발현·구조·조건을 작은 공통 특징으로 결합 | encoder 폭 비교 미실행 |
| hidden128 | 간단한 비선형 잔차 예측기 | 64/256보다 좋은지 미검증 |
| learning rate0.0003 | 초기 학습률로 고정 | 이 프로토콜에서 최적성 미검증 |
| AdamW weight decay0.01 | 가중치 정규화 | 효과·적정 강도 미검증 |
| batch64, clip1 | 제한된 계산 예산과 gradient 제어 | 민감도 미검증 |
| cosine30step,16sample | 생성 계산량 제한과 평균 예측 | 단계·sample 수 최적성 미검증 |
| max100epoch | 초기 반복 평가의 학습 예산 | 1/25/50/75/100 중 내부 선택. 상한 선택 조건도 존재 |

가중치 parameter는 학습되지만 위 숫자는 사람이 지정한 hyperparameter입니다. 초기값의 목적을 설명할 수 있다는 것이 최적성의 증거는 아닙니다. dropout은 이 모델에 없습니다.

## 이번에 작성한 검증 코드

`scripts/validate_ddpm_architecture.py`는 **외부 시험 성능을 계산하지 않는** 내부 검증 민감도 실행기입니다. 연구 데이터에서는 아직 실행하지 않았습니다.

우선 network 폭만 한 번에 하나씩 바꿉니다.

| 후보 | context | hidden |
| --- | ---: | ---: |
| reference | 64 | 128 |
| context32 | 32 | 128 |
| context128 | 128 | 128 |
| hidden64 | 64 | 64 |
| hidden256 | 64 | 256 |

reference와 비교하는 기준은 model seed별 대응 검증 RMSE 차이입니다. 다른 PCA·loss·optimizer·학습률·생성 조건은 고정합니다. 모든 후보에 같은 epoch·λ 선택 기회를 줍니다. λ0이면 기본 Ridge만 사용해 검증 기준선에 못 미치는 보정을 강제로 채택하지 않습니다.

기본은 split seed42, 외부5-fold 중fold0에서 내부 학습89개·검증23개, model seed42/43/44입니다. 후보5개×seed3개=15회 내부 fit입니다. 선택 후112개 refit이나 외부 테스트 평가를 이번 코드에 넣지 않았습니다. 데이터 로더는 전체 배열을 읽지만, 외부 테스트 정답은 적합·점수 계산·후보 선택에 사용하지 않습니다. 입력·출력 PCA 및 기준선도 내부 학습에만 적합합니다.

모델 seed마다 검증으로 epoch·λ를 고른 후, 세 seed의 검증 RMSE 평균이 가장 낮은 후보를 기록합니다. 이 검증 평균은 튜닝에 사용한 점수이므로 새 화합물 성능의 독립 추정이 아닙니다. seed는 같은 검증 화합물을 공유하므로 독립 화합물 표본으로 세지 않습니다.

## 실행과 산출물

```bash
python scripts/validate_ddpm_architecture.py \
  --prepared /home/seowon1211/data1/conditioned_expression/prepared_controls \
  --structure /home/seowon1211/data1/conditioned_expression/morgan_matched/morgan_binary.csv \
  --out /새로운/validation_width_폴더
```

기본은 계획만 출력합니다. 실제 내부 학습에는 `--execute`를 붙입니다. source/input hash·환경·후보·fold를 run.json에 기록합니다. 완료 trial은 identity와 파일 hash를 확인해 건너뛰고, 미완료는 새로운 attempt로 재시작합니다. 기존 연구 결과와 원본 실행 셀은 수정하지 않습니다.

- candidate_summary.csv: 후보별 validation RMSE mean/std와 parameter 수.
- trials.csv: seed별 선택 epoch·λ와 reference 대비 대응 차이.
- selected_candidate.json: 다섯 후보 안에서 선택된 설정. 전역 최적값이라는 뜻은 아님.
- compound_split.csv: 내부 학습·검증·외부 시험(평가 안 함) 목록.
- trial별 validation_history.csv·selection.json·best_validation_model.pt: 검증 기록. λ0 유지 시 checkpoint가 없을 수 있음.

## 다음 판단의 경계

이 한 번의 실험으로 PCA차원·학습률·step·sample 수까지 검증됐다고 말하면 안 됩니다. network 폭의 작은 변화에도 결과가 비슷하면 초기값이 적어도 이 범위에서 민감하지 않았다고 설명할 수 있습니다. 특정 후보가 좋더라도 이 검증집합에서의 선택입니다.

외부5-fold로 튜닝 모델의 성능을 보고하려면 **각 outer fold 안에서 별도로 후보를 선택한 뒤 해당 fold만 평가하는 nested 절차**가 필요합니다. 다른 outer fold들의 내부 검증 결과를 합쳐 전역 설정을 고른 후 같은5-fold 테스트를 독립 성능처럼 보고하면 안 됩니다. 이미 연구에서 관찰한140개 자료이므로 전체 절차도 탐색적이라는 한계는 남습니다.

조건별 validation score를 보고 후보 범위를 계속 바꾸면 검증집합에도 적응할 수 있습니다. 시작 전에 후보와 비교 목적을 고정하고 결과에 따라 실험 이력을 분리하세요. 결과를 보고 기존 cGAN/MLP와 비교할 때는 DDPM만 추가 튜닝한 버전임을 표시해야 합니다.

## 코드 검사

합성 검사4개에서 계획 모드의 무실행, 기준 network 초기 가중치의 기존 구현과 완전 일치, 한 번에 한 폭만 바뀜, 축소한 합성 검증과 재개, 외부 시험 정답을 크게 바꿔도 후보 검증 결과가 변하지 않음을 확인했습니다. 실제 연구 데이터 최적성이나 재현 성능 검증은 아직 아닙니다.
