# 완료한 반복실험에 Descriptor만 추가

`repeat_time_structure_pilot.py --arms descriptor --reference-run EXISTING_RESULTS`는 기존 기본/Morgan 결과를 보존하고 Descriptor × MLP/cGAN/DDPM만 추가한다. 5fold × model seed3 × 3계열 =45조건, 최대90fit. 최대100epoch/동일분할·모델seed·source슬롯·PCA32·모델폭·optimizer·검증/재학습 정책을 유지한다.

## 앞선 이상 신호와 전처리 변경

단일holdout의 원래 Descriptor는 내부 검증RMSE가24~30으로 다른입력약0.31보다 매우 컸다. 극단값이나 작은학습분산의 증폭이 가능한 원인이지만 원자료/preprocess artifact를 독립검사해 원인을 확정한 것은 아니다. 따라서 기존 Descriptor점수를 정상적인 비교로 재사용하지 않는다.

추가실험은 `bounded_descriptors.py`의 고정된 수치안정화 정책을 사용한다. 테스트성능으로 clipping범위나 변환을 최적화한 것이 아니다.

1. 저장된 SMILES에서 2D Descriptor를 계산한다. PubChem 재검색·Morgan 재계산 없음.
2. 고유 학습 화합물에서 비유한값이 있는 열을 제거한다.
3. `sign(x)*log1p(abs(x))`를 적용해 큰분포꼬리를 줄인다. **단순 원시 Descriptor의 표준화와는 다른 표현**이다.
4. 학습 변환값의 상수/기계정밀도 수준의 거의상수열을 제거한다.
5. 변환한 학습값의 abs(Pearson)>0.95 중복열을 이름순 결정규칙으로 제거한다.
6. 학습 MinMaxScaler(feature_range=(-1,1),clip=True)를 적합한다.
7. 검증/테스트의 선택열결측은 변환공간의 학습중앙값으로 채우고, 학습범위를 벗어난 값은 [-1,1]로 제한한다. test로 scaler나 범위를 재적합하지 않는다. 분자전체부재는0+presence0이다.
8. 비유한입력 또는 경계초과가 남으면 중단한다.

모든학습범위·변환·선별은 내부학습/외부재학습에서 각각다시적합한다. 내부/최종 `*_descriptor_numeric_audit.csv`에 target화합물별선택열결측수와 clipping수를 남긴다. clipping이 많다면 학습범위외 화학적특징을 구별하지 못하는한계가 크므로 반드시 보고한다. 경계처리는 모델입력의수치폭주를막는장치이지 화학구조의동일성이나 일반화를검증하는것이 아니다.

## 기존 결과와 비교하기 위한 검사

참조 폴더는 앞선반복실험의 `results`폴더다. 해당폴더는완료상태여야하며 다음을검사한다.

- prepared/Morgan/SMILES의경로와파일hash
- 화합물fold수/분할seed/모델seed목록/family/epoch상한/threads
- 각fold·seed의공통학습설정(arm과Descriptor정책만예외)
- 공유 core/기존화학특징함수/모델/로더snapshot/평가코드hash
- Python/RDKit/numpy/pandas/torch/sklearn/scipy/joblib 버전
- 추가실행의각fold·seed 테스트target ID와시간×등급기준선예측이참조와정확히동일한지

pilot실행기의Descriptor분기와추가wrapper는기능확장으로달라진부분이며공유핵심모델·입력코드와구분한다. 이전학습을다시실행하지않고기존요약CSV를재사용한다. 과거예측에서기존점수를독립재계산한검증은아니다. 재사용한CSV의SHA256을reference_reuse.json에보존한다.

일치하지않으면중단하며무리하게설정을변경해병합하지않는다. 참조없이새Descriptor실험만실행할수도있으나비교가능성을자동보증하지않는다. 옛실행폴더나코드snapshot은수정하지않는다. 새코드로옛폴더를resume하지않는다.

## 출력

새실험의summary.csv는Descriptor와기준선만포함한다. 전체완료·참조검사통과후:

- combined_summary.csv: 기존6조건+Descriptor3조건+기준선
- combined_classification_summary.csv: 동일조건의탐색적F1/방향지표
- descriptor_paired_differences.csv: 같은모델seed의Descriptor−basic, Descriptor−Morgan
- reference_reuse.json: 참조경로/정책/재사용파일hash

원래 results/fold_N/seed_S 아래결과는읽기만한다. 기존표를덮어쓰지않는다. OOF target조건전체를합쳐평가하며fold별RMSE를단순평균하지않는다. 모델seed표준편차는초기화변동만의미한다. 테스트를관찰한뒤추가한전처리·표현비교이므로탐색결과로보고한다.

기본은plan-only. `--execute`에서만학습한다. 전역60분/개별15분소프트예산과동일명령재개정책은기존반복실험과같다. 출력폴더는새로지정한다. cloud에서는극단값·결측·train-only선별·참조보존·자동병합을합성데이터로검사했고실제연구자료학습은실행하지않았다.
