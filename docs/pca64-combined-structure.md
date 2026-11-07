# 입력 PCA64와 두 화학구조 표현의 결합

발현 입력 PCA64 / 출력 전체 probe를 고정하고 basic, morgan, descriptor, both를 MLP/cGAN/DDPM에서 비교한다. both는 Morgan과 RDKit 2D 기술자의 결합이며 pathway 추가가 아니다. 64는 한 내부 분할의 구조 없는 MLP 검증에서 선택한 값이고 모든 모델에 대한 최적성을 주장하지 않는다.

`meeting_time_structure_pilot.py --input-components 64 --input-space pca --output-space full --structure-layout separate --arms basic morgan descriptor both`로 실행한다. 기본 차원32와 shared 레이아웃은 이전 실행 호환을 위해 유지한다. both는 separate를 요구한다.

separate 레이아웃은 같은 폭의 Morgan 블록과 descriptor 블록을 전 조건에 예약한다. 미사용 블록은 모두 0이다. 활성 블록에는 각 표현의 presence flag가 있다. 모델 계열 안에서 구조 조건별 입력 크기와 명목 파라미터 수는 동일하지만 유효 정보와 활성 파라미터는 다를 수 있다. 이전 shared 레이아웃 결과와는 모델 입력 크기가 달라 직접 섞지 않는다.

MinMax/PCA/score 표준화 및 기술자 선별/스케일은 내부 학습으로 적합한다. 내부 검증 RMSE로 epoch를 선택하고 외부 학습 전체로 전처리와 모델을 다시 적합한 뒤 보류 화합물을 평가한다. 출력은 학습 평균/공통 RMS만 변환하고 probe를 압축하지 않는다. 단일 분할12조건은 최대24회 fit이며 반복 교차검증이 아니다. max100epoch와 나머지 기존 모델 설정을 유지하며 100epoch 선택이 수렴을 증명하지 않는다.

저장 SMILES에서 descriptor만 계산한다. PubChem 재검색, Morgan 재생성, 원자료 다운로드는 하지 않는다. run.json 입력/소스 hash, 분할, 모델 크기, 전처리, 검증 기록, 예측 및 평가를 보존한다. 무변화 및 훈련 화합물 시간×등급 평균 반응 기준선을 포함한다. 같은 테스트를 이전 설계에서 관찰했으므로 탐색 평가다.

합성 데이터 검사: 전체 probe 출력12조건 실행, 구조 블록 결합/0 마스킹/동일 입력 크기, held-out 기술자가 학습 변환에 영향이 없는지 검사. 실제 서버 학습 재현 검증과 구분한다.
