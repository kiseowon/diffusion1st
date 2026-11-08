# 입력 PCA64와 두 화학구조 표현의 결합

발현 입력 PCA64 / 출력 전체 probe를 고정하고 basic, morgan, descriptor, both를 MLP/cGAN/DDPM에서 비교한다. both는 Morgan과 RDKit 2D 기술자의 결합이며 pathway 추가가 아니다. 64는 한 내부 분할의 구조 없는 MLP 검증에서 선택한 값이고 모든 모델에 대한 최적성을 주장하지 않는다.

`meeting_time_structure_pilot.py --input-components 64 --input-space pca --output-space full --structure-layout separate --arms basic morgan descriptor both`로 실행한다. 기본 차원32와 shared 레이아웃은 이전 실행 호환을 위해 유지한다. both는 separate를 요구한다.

separate 레이아웃은 같은 폭의 Morgan 블록과 descriptor 블록을 전 조건에 예약한다. 미사용 블록은 모두 0이다. 활성 블록에는 각 표현의 presence flag가 있다. 모델 계열 안에서 구조 조건별 입력 크기와 명목 파라미터 수는 동일하지만 유효 정보와 활성 파라미터는 다를 수 있다. 이전 shared 레이아웃 결과와는 모델 입력 크기가 달라 직접 섞지 않는다.

MinMax/PCA/score 표준화 및 기술자 선별/스케일은 내부 학습으로 적합한다. 내부 검증 RMSE로 epoch를 선택하고 외부 학습 전체로 전처리와 모델을 다시 적합한 뒤 보류 화합물을 평가한다. 출력은 학습 평균/공통 RMS만 변환하고 probe를 압축하지 않는다. 단일 분할12조건은 최대24회 fit이며 반복 교차검증이 아니다. max100epoch와 나머지 기존 모델 설정을 유지하며 100epoch 선택이 수렴을 증명하지 않는다.

저장 SMILES에서 descriptor만 계산한다. PubChem 재검색, Morgan 재생성, 원자료 다운로드는 하지 않는다. run.json 입력/소스 hash, 분할, 모델 크기, 전처리, 검증 기록, 예측 및 평가를 보존한다. 무변화 및 훈련 화합물 시간×등급 평균 반응 기준선을 포함한다. 같은 테스트를 이전 설계에서 관찰했으므로 탐색 평가다.

합성 데이터 검사: 전체 probe 출력12조건 실행, 구조 블록 결합/0 마스킹/동일 입력 크기, held-out 기술자가 학습 변환에 영향이 없는지 검사. 실제 서버 학습 재현 검증과 구분한다.

## 고정 설정 반복 검증

`repeat_time_structure_pilot.py`에 같은 `--input-components 64 --structure-layout separate --output-space full --arms basic morgan descriptor both`를 전달한다. 화합물5-fold × 모델 초기화42/43/44 ×12조건 =180조건, 내부학습 및 최종 재학습 최대360fit이다. fold0 모델seed42 파일럿은 새 실행에 다시 계산하며 이전 폴더를 덮어쓰거나 결과를 합치지 않는다.

1차 지표는 전체 out-of-fold target/probe 오차를 합친 response RMSE다. 구조 추가−없음과 both−각 단독, 같은 구조에서 DDPM−MLP/cGAN을 같은 seed로 비교한다. 시간×등급 평균 및 무변화 기준선도 포함한다. F1/MAE/상관은 보조 지표로 함께 보고한다. 일부 fold만 끝난 결과를 전체 성능으로 표시하지 않는다. 모델seed별 전체 OOF 성능을 먼저 계산하고 평균/표준편차를 보고하며 예측 ensemble은 하지 않는다. 기준선을 모델seed 수만큼 독립 결과로 세지 않는다.

분할 seed는42 하나이며 같은5-fold를 세 모델seed에서 재사용한다. 따라서 독립적인 화합물 분할3회가 아니며 표준편차는 모델 초기화 및 생성 난수 변동을 나타낸다. 모델간 생성목적함수/튜닝범위가 동일하지 않다. PCA64가 한 내부분할 MLP에서 선택됐고 과거 외부테스트를 관찰한 한계도 유지한다. 독립 확증시험/전체 모델의 최적 파라미터 검증/화합물 bootstrap CI라고 표현하지 않는다.

기존 단일 실행의100epoch 상한을 그대로 고정하며 결과 도중 설정을 조정하지 않는다. 상한이 선택되는 경우 그대로 보고한다. 재개는 같은 출력·입력·코드·환경·옵션에서만 허용된다. 최신 코드로 갈아끼워 옛 출력에 재개하지 않는다. 변경된 차원/레이아웃은 계약검사에서 거부된다. 실제 서버 학습은 사용자가 수행하며 합성24조건·재개·계약불일치 검사를 실제 재현 완료로 표시하지 않는다.

## 저장된 입력의 범위 진단

`scripts/inspect_saved_input_ranges.py --unit /RUN/results/fold_1/seed_43 --out /NEW_REPORT`는 저장된 basic 전처리를 읽고 fit 없이 source 차분→MinMax→PCA→score 표준화를 재적용한다. source/target 용량·시간, 미관측 단위/등급, 학습 때 미관측 slot을 기록한다. 학습·모델 추론·예측 재생성은 하지 않는다. 입력 hash와 sklearn 버전을 대조하며 원래 결과/전처리를 수정하지 않고 새로운 진단 폴더에만 기록한다. 내부와 최종 변환, 모든 partition을 CSV에 보존한다.

범위 이탈은 원인 입증이 아니다. 용량의 수치 비교는 물리 단위 환산이 아니다. 저장 모델의 동작이나 구조 특징을 진단한 것으로 해석하지 않는다. 합성 검사에서 기존 파일 hash 불변·예측/학습 호출 없음·변경 입력 거부를 확인했다.
