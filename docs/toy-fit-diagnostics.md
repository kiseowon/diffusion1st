# 고정 toy 모델의 학습집합·테스트 예측 진단

`diagnose_toy_fit.py --run /TOY/results --out /NEW_REPORT`는 저장된 최종 모델과 전처리만 사용한다. 재학습·전처리 fit은 하지 않지만 **예측은 다시 생성**한다. 원래 입력/공통 코드 hash·주요 라이브러리 버전을 확인하고, 같은 원래 test noise seed(model_seed+30000)로 저장된 테스트 예측을 먼저 재현한다. 기존 수치 재현 검사 기준을 통과해야 진행하며 통과 여부와 실제 차이를 모두 보존한다. 원본 파일은 변경하지 않는다.

## 입력과 정답

현재 toy는 입력도 in vitro 처리군−같은 화합물·단위·시간의 대조군 차이다. 이 차이에 저장 MinMax→PCA64→score 표준화를 적용한다. 정답은 in vivo 처리군−대응 대조군이다. 모델 내부 target의 학습 probe 평균 중심화/공통 RMS 변환은 이 생물학적 차분을 계산한 **뒤** 적용한 별도 변환이다. 실제 테스트 in vivo 대조군은 정답 반응 계산에만 사용하며 생성 입력에 넣지 않는다.

## 서로 다른 평가를 구분

- final_training: 최종 재학습에 사용한 내부학습+이전 내부검증 화합물. 최종 모델에 대한 in-sample 평가다.
- test: 원래 보류 화합물. 같은 최종 모델의 out-of-sample 평가다.
- selected_inner_validation_rmse: 다른 모델/전처리 단계에서 epoch를 선택할 때 기록된 값이다. 이를 최종 모델의 검증 성능이라고 부르지 않는다.
- ddpm_training_denoising: 최종 학습 정답에 고정 t의 노이즈를 섞어 원래 반응을 복원하는 teacher-forced 진단이다. 학습 정답을 noisy input에 쓰므로 자유 생성 예측 성능과 같은 지표가 아니다. 테스트 정답은 이 진단에 사용하지 않는다. t=0/5/10/15/19에서 표준화 x0 MSE와 실제 probe 공간 복원 RMSE를 기록한다.

훈련/테스트 모두 동일한 최종 학습 평균 반응 및 무변화 기준선과 비교한다. 원래 생성4회 평균과 noise seed를 유지한다. split별 baseline 오차가 다를 수 있으므로 raw train-test gap만으로 과적합을 입증하지 않는다. skill_vs_training_mean=1−model_MSE/baseline_MSE는 해당 split에서 학습 평균 반응 기준선 대비 개선을 나타내며 기존 R²와 구분한다.

화합물 간 예측 변동 RMS와 정답 변동 RMS의 비율도 보존한다. 낮으면 평균적 프로파일에 가까운 예측이라는 단서지만 조건을 완전히 무시한다는 증거는 아니다. 생성모델의 변동에는 생성 난수도 포함된다. 하나의 샘플링 seed/생성4회 평균 진단만으로 생성 분포의 품질을 주장하지 않는다.

## 출력

- summary.csv / fit_metrics.csv: 모델·초기화·phase별 RMSE/MAE/Pearson, 기준선 대비 skill, 화합물 간 변동.
- by_compound.csv: 화합물별 오차.
- ddpm_training_denoising.csv: teacher-forced 학습 노이즈 복원 진단.
- classification.csv: 각 phase의 고정 임계값 분류 지표.
- 모델별 원본 재현 검사 JSON, 학습/테스트 진단 예측 NPZ, provenance.json의 소비파일/진단코드 hash.

최종 학습 화합물에서도 기준선보다 나쁘면 현 checkpoint가 반응 차이를 충분히 학습하지 못했다는 단서다. 이것만으로 데이터에 예측 신호가 없거나 optimizer가 실패했다고 단정하지 않는다. 학습에서는 좋고 테스트에서 나쁘면 일반화 문제의 단서지만 분포 차이도 검토한다. 노이즈 복원이 좋고 자유 생성이 나쁘면 역과정/조건 활용/표본 평균 등 추가 분석이 필요하며 원인을 단정하지 않는다.

합성 테스트는 원본 test 예측 재현, 최종학습/테스트 집합 구분, 학습/전처리 fit 호출 없음, teacher-forced 진단의 학습 정답만 사용, 원본 불변을 확인한다. 실제 서버 진단 완료와 구분한다.
