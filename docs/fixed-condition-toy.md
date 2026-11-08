# 24시간·High 고정 조건 toy 실험

한 화합물의 in vitro 24h·High 발현 반응 하나로 in vivo 24h·High 반응 하나를 예측한다. 여러 시간의 source 연결이나 pair 평균을 사용하지 않는다. AIVIVE가 RMA 처리해 배포한 자료의 조건 평균과 처리군−대조군 차이를 사용하며, 이 코드에서 RMA나 추가 log 변환을 하지 않는다. 원래 결과는 보존하고 새 폴더에서 수행한다.

## 공통 화합물 집합

다음 조건을 모두 만족하는 화합물 집합을 모델 평가 전에 확정한다.

- source 24h·High와 target 24h·High가 각각 정확히 한 조건.
- 같은 화합물·단위·시간의 대응 대조군 존재.
- 저장된 Morgan과 파싱 가능한 저장 SMILES의 2D 기술자 존재.

같은 High 관측이 여러 개이면 임의로 선택하지 않는다. 구조 없는 모델에도 동일한 집합을 사용한다. 파싱 가능하며 최소 한 유한 기술자가 있다는 규칙이므로, 모든 개별 기술자 값이 완전하다는 뜻은 아니다. target 반응 크기나 모델 성능으로 포함 여부를 정하지 않는다. 실제 포함 수와 제외 사유는 cohort_audit.csv에 기록하며 140/137개로 미리 단정하지 않는다.

**High는 화합물별 실험 등급이며 동일한 실제 농도·투여량이 아니다.** 단위를 환산하지 않으며 정량적 용량 외삽을 평가하지 않는다. 원래 용량·단위는 cohort.csv에 보존하지만 모델에는 입력하지 않는다. 정답이 있는 조건만 예측하며, 실제 target 반응이나 target 대조군 발현은 생성 입력에 넣지 않는다. source 대조군은 source 반응 계산에 사용한다.

## 9조건과 전처리

구조 없음 / Morgan / RDKit 2D 기술자 × MLP / cGAN / x0-DDPM을 비교한다. Morgan은 기존 binary를 읽고, 기술자는 저장 SMILES에서 계산한다. 외부 PubChem 검색이나 Morgan 재생성은 하지 않는다. RAID의 alvaDesc 854개를 재현하는 실험은 아니다.

입력은 학습 source만으로 MinMax → PCA64 → score 표준화를 적합한다. 64는 이전 실험에서 가져온 초기 설정이며 새 코호트의 최적 차원을 검증한 것은 아니다. 학습 수나 유효 rank가 부족하면 자동으로 차원을 바꾸지 않고 중단한다.

출력은 전체 3,453개 probe다. 학습 target의 probe별 평균을 빼고 공통 RMS로 나눈 뒤 예측에서 역변환한다. 출력 PCA와 probe 필터는 없다. 기술자는 학습 화합물에서 비유한·상수·고상관 열을 제거하고 signed log → 학습 MinMax[-1,1] clip을 적용한다. 테스트의 선택 열 결측은 학습 중앙값으로 대체하며 결측·clip 수를 기록한다.

세 구조 입력의 폭을 padding으로 맞추고 구조 없음에는 0을 넣는다. 공통 집합 모두 구조가 있으므로 presence flag는 추가하지 않는다. 같은 모델 계열 내 명목 파라미터 수는 같지만 활성 입력과 유효 자유도가 같다는 뜻은 아니다.

## 작은 모델의 초기 설정

발현 PCA → 64 SiLU와 구조 벡터 → 32 SiLU를 별도 encoder로 처리하고, 결합 96 → 64 SiLU로 context를 만든다.

- MLP: context64 → 128 → 전체 probe.
- cGAN: context + noise16 → 128 → 전체 probe. 판별기는 별도 context encoder와 실제/생성 발현 → 128 → logit.
- DDPM: context + noisy 전체 probe + time embedding16 → 128 → 전체 probe의 x0 예측.

기본 PyTorch 초기화, Adam1e-4·weight_decay0, batch32, clip1, 최대100epoch를 사용한다. cGAN 판별기는 SGD1e-4·momentum0.9, 생성기 손실은 10MAE+2BCE다. MLP는 MAE, DDPM은 x0 MSE다. cosine20step·생성4회 평균을 사용한다. encoder는 예측 모델과 함께 학습하며 VAE·Ridge 보정·KEGG·FiLM·graph는 사용하지 않는다.

이 값은 규모를 제한한 초기 선택이며 최적성이나 모델별 충분한 튜닝을 입증하지 않는다. 모델 계열 간 파라미터 수는 다르며 실제 수를 저장한다. 생성모델도 조건 평균의 점 예측을 평가하므로 개별 동물 분포의 품질을 검증하지 않는다. 이전 실험과 코호트·모델이 달라 과거 RMSE와 직접 비교하지 않는다.

## 분할과 평가

포함 코호트를 먼저 확정한 뒤 seed42로 5-fold를 정의하되 fold0만 평가한다. 외부 학습의 20%를 내부 검증으로 나눠 원래 probe 공간의 RMSE로 epoch를 선택한다. 이후 외부 학습 전체에서 전처리를 다시 적합하고 모델을 초기화해 선택 epoch만큼 재학습한다.

기본 model seed42/43/44는 같은 분할에서 초기화와 생성 난수를 바꾼다. **27조건·최대54fit이며 전체 5-fold 교차검증이 아니다.** 코호트가 달라 이전 fold0와 테스트 화합물도 다를 수 있다. 원자료를 이전에 관찰한 이력이 있으므로 새로운 독립 확증시험으로 표현하지 않는다.

무변화와 외부 학습 화합물의 probe별 평균 반응을 기준선으로 사용한다. 화합물당 target이 하나이므로 동일 target 가중은 동일 화합물 가중이기도 하다. RMSE가 1차 지표이고 MAE·Pearson·Spearman·R²·고정 임계값0.5/1.0 F1는 보조다. 모든 조건·seed를 보존하며 최저 결과만 선택하거나 seed 평균 지표와 예측 ensemble을 혼동하지 않는다. 유의성 검정은 제공하지 않는다.

## 실행과 기록

`python scripts/fixed_condition_toy.py --prepared /PREPARED --structure /MORGAN.csv --smiles /SAVED_SMILES.csv --out /NEW_RESULTS --execute`

--execute 없으면 계획만 표시한다. 입력·코드 hash와 라이브러리 버전, 코호트·분할·control audit·전처리·설명 분산·모델 크기·검증 기록·모델·예측·평가를 저장한다. 같은 코드·환경·입력·설정·경로로 재개하면 완료 조건을 검사해 재사용하고, 변경된 설정으로 같은 출력에 재개하는 것은 거부한다.

주요 파일은 summary.csv, paired_rmse.csv, classification_summary.csv다. 합성 테스트는 공통 집합 규칙, held-out 전처리 격리, 9조건 실행, 같은 계열의 명목 모델 크기, 재개·불일치 거부를 검사한다. 실제 서버 학습 재현은 아직 수행하지 않았다.
