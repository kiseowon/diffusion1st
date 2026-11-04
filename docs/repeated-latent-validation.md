# 고정된 저차원 잔차 모델의 반복 검증

이번 실행기는 서버에서 한 번 비교한 잠재 MLP/x₀-DDPM의 설계를 별도 CLI로 연결한 **새 반복 평가 코드**입니다. 보존한 과거 실행 셀은 수정하지 않습니다. 연구 데이터 학습은 코드 작성 환경에서 실행하지 않았고, 검사에는 작은 합성 데이터만 사용했습니다.

## 고정 비교와 비용

- 기본: split seed42의 화합물5-fold × model seed42/43/44 × MLP/DDPM.
- 총15개 fold·초기화 조합,30개 모델 조건. 내부 학습과 선택 후 refit을 포함해 **최대60회 신경망 fit**입니다. 보정0이면 최종 refit은 생략합니다. Ridge fitting은 이 숫자에 포함하지 않습니다.
- 확장: `--split-seeds 42 43 44`는45개 fold·초기화 조합,90모델 조건,최대180회 신경망 fit입니다. 확장하기 전에 1차 결과를 평가하고, 설정 변경 실험과 고정 설정 반복을 구분하세요.
- 140화합물이면 외부 fold당28개, 내부 학습89·검증23·refit112입니다. 화합물을 정렬→split seed로 순열→순서 modulo5로 외부 fold를 배정합니다. 원래 단일28개 테스트와 같은 분할을 재사용하는 것이 아닙니다.
- 각 외부 학습 집합의20%(올림)를 화합물 단위 내부 검증으로 분리합니다. 내부 분할은 split seed·fold로 결정하며 모든 모델·model seed가 공유합니다.

## 유지한 설계

입력은 source의 matched-control 변화량, source/target dose·time·unit·level, Morgan binary+presence입니다. source 특징은 target별로 먼저 평균합니다. source scaler와 조건 변환을 학습 pair/source만으로 적합하고, 학습 target의 source 평균 특징으로 입력PCA 최대64축과 score scaler를 적합합니다. 거의0인 PCA축은 상대 singular tolerance1e-8로 제외합니다.

기본 예측은 목표 조건만의 Ridge입니다. 내부 검증에서 α10/100/1000을 선택합니다. Ridge의 **훈련 내 예측** 잔차를 학습 자료에서 PCA 최대32축으로 압축하고, 학습 score 전체의 RMS 한 개로 나눕니다. 이번 반복 코드도 OOF 잔차 방식으로 바꾸지 않습니다.

MLP는 noisy latent0·time0으로 잠재 정답을 직접 예측합니다. DDPM은 noisy latent와t를 받아 x₀ MSE를 학습합니다. network는 입력→64context, time embedding16, 결합→128hidden→latent output입니다. cosine30step 역과정,16sample 평균,별도 clipping 없음. MLP와 DDPM은 같은 입력·PCA·기준선·내부 검증을 사용하나 유효 자유도·추론 연산량은 같지 않습니다.

AdamW lr0.0003,weight_decay0.01,batch64,clip1,최대100epoch를 고정합니다. epoch1/25/50/75/100과 λ0/0.1/0.25/0.5/1을 내부 검증 반응RMSE로 선택합니다. λ0을 초기 후보로 포함합니다. 선택 후112개로 전처리·Ridge·PCA를 다시 적합하고 새 모델을 선택epoch만큼 학습합니다. 실제 probe 반응 공간으로 역변환한 뒤 보정 전체에λ를 곱합니다.

validation 생성 seed=model seed+20000, test seed=model seed+30000입니다. model seed 반복은 초기화·훈련난수·생성난수가 함께 달라지는 반복입니다. 이 변동의 원인을 각각 분리하는 실험은 아닙니다. pathway·AE·VAE·graph·새 hyperparameter 탐색은 포함하지 않습니다.

## 기준선

모든 외부 fold에서 다음을 함께 계산합니다.

1. 목표 dose(log1p)·time(표준화)·unit/level(one-hot)의 Ridge.
2. 시간×용량 등급 평균 반응: 고유 훈련 target만 사용→화합물별 평균→화합물 간 평균. 원래 단위 간 물리 환산이나 시간까지의 보간을 추가하지 않습니다. 시험 group이 훈련에 없으면 임의 fallback 없이 실패합니다.
3. 반응0 기준선.

기준선의 실제 테스트 대조군은 정답 반응 계산에만 사용합니다. 모델은 반응을 직접 보정하며 absolute RMSE를 만들지 않습니다. 화합물마다 조건 수가 달라도 주 지표는 target×probe 균등 RMSE로, compound macro-average가 아닙니다.

## 실행과 재개

필요한 입력은 prepared_controls의 발현NPY2개,조건CSV2개,pairs.csv.gz,probes.csv와 morgan_binary.csv입니다. KEGG annotation은 이번 실행에 필요하지 않습니다. 로더는 저장소에 보존한 first_structure_mlp.py snapshot을 사용합니다.

```bash
python scripts/repeated_latent_validation.py \
  --prepared /home/seowon1211/data1/conditioned_expression/prepared_controls \
  --structure /home/seowon1211/data1/conditioned_expression/morgan_matched/morgan_binary.csv \
  --out /새로운/결과/폴더
```

기본은 계획 출력만 하며 입력 읽기·폴더 생성·학습이 없습니다. 실제 실행에는 같은 명령에 `--execute`를 붙입니다. CPU만 지원합니다. 현재 Python환경에 numpy,pandas,scipy,scikit-learn,torch,joblib,threadpoolctl이 필요합니다. 패키지를 자동 설치하거나 연구 자료를 다운로드하지 않습니다.

재개는 **같은 코드 경로·입력·out 경로·모든 인자와 환경**으로 실행합니다. run.json은 source/loader와 입력 hash,설정,라이브러리 버전을 기록합니다. 완료 조건은 complete.json의 job identity와 산출물 hash가 일치할 때만 건너뜁니다. 미완료 시 새 attempt를 만들고 이전 실패파일은 보존합니다. epoch 도중 optimizer 상태부터 재개하는 기능은 아닙니다. fold의 기준선과 전처리는 같은 계약 아래 다시 적합할 수 있습니다.

## 결과 읽기

| 파일 | 의미 |
| --- | --- |
| compound_folds_*.csv, split_*/fold_*/compound_split.csv | 외부·내부 화합물 분할 |
| run.json, *_audit.csv | 실행 계약·대조군 연결·구조 누락 |
| split_*/fold_*/baseline_predictions.npz, ridge.joblib, *_preprocess.joblib | 기준선과 단계별 전처리 |
| 모델별 complete.json | 유효 완료 attempt와 hash를 가리킴 |
| attempt_*/validation_history.csv,selection.json,model.pt,predictions.npz,condition_metrics.csv | 실제 선택·예측 산출물. λ0이면 model.pt 없음 |
| results.csv | 조건별 최종 지표 |
| pooled_by_split_model_seed.csv | fold RMSE 평균이 아니라 전체OOF target 오차를 합산한 지표 |
| paired_comparisons.csv | 같은 split/model seed의 DDPM−MLP 및 기준선 대비 차이 |
| summary.csv | target별 seed·repeat **MSE**를 먼저 평균한 후 pooled RMSE. 예측 ensemble이 아님 |
| compound_bootstrap.csv | 아래 범위의 화합물 단위 재표집 percentile 구간 |
| completed.json | 전체 기대 조건 수가 완료됐을 때 생성 |

baseline은 model seed 수만큼 중복 집계하지 않습니다. 여러 split의 반복에서도 화합물이나 target이 새로 추가되는 것은 아닙니다. primary paired difference는 DDPM−MLP이며 음수이면 DDPM 오차가 낮습니다. 기준선 대비 차이도 함께 보고합니다. 개선 횟수만으로 유의성을 판단하지 않습니다.

기본1000회 bootstrap은 각 target의 반복/seed MSE를 먼저 평균한 뒤 화합물 단위로 재표집하고, 각 화합물의 모든 target을 함께 보존합니다. **이미 학습된 고정 모델의 평가 화합물 구성 변동을 보는 탐색적 구간**입니다. 매 bootstrap마다 재학습하지 않으므로 학습집합 변동·튜닝 불확실성·cross-validation 의존성 전체를 포괄하는 엄밀한 일반화 신뢰구간은 아닙니다. p-value는 만들지 않습니다. `--bootstrap-samples 0`으로 생략할 수 있습니다.

## 해석과 검증 상태

모든 조건은 구조를 켜므로 구조 추가 효과를 검증하지 않습니다. 단일 분할seed로 시작하며 모든140개 화합물은 이미 연구 과정에서 관찰한 집합입니다. seed42 단일 테스트 결과0.310624를 독립적으로 재확증한 시험이 아니고, 완전히 새로운 최종 시험이라고 부르지 않습니다. 결과를 본 뒤 프로토콜을 바꾸면 새로운 탐색 실험으로 분리합니다.

합성 검사에서는 계획 모드의 무실행,화합물 비중첩,학습 전처리의 held-out 불변성,훈련 정답에 held-out latent를 넣지 않음,두 모델의 생성·역전파,기준선 화합물 균등 평균,축소한 합성 전체 흐름·재개·설정 불일치·산출물 변조 거부,seed 집계 방식을 확인했습니다. 이는 실제 연구 결과 재현이 아닙니다. 검사 환경의 PyTorch는2.15.0+cpu이며 원래 서버의2.5.0+cu121과 같다고 주장하지 않습니다. 실행환경은 각 run.json에 새로 기록합니다.

## cGAN 확장

`--families mlp cgan ddpm`으로 같은공통설정의cGAN을추가할수있습니다. 기본은mlp/ddpm을유지합니다. [세계열비교설계](three-family-comparison.md)를참고하세요. 새세family계획은45조건·최대90fit이며원래30조건결과를덮어쓰지않습니다.
