# MLP 입력 정보의 기여도: 목표 조건·in vitro·Morgan

새 모델을 추가하기 전에 미관측 화합물 예측에서 입력 정보의 기여를 비교한다. 기본은 화합물5-fold, 분할seed42, 모델seed42/43/44, 출력전체3453 probe, 기존입력PCA64/slot레이아웃과 MLP(context256/head64, Adam1e-4, batch128, 최대100epoch, 내부검증RMSE로epoch선택)를 유지한다. 선택규칙은 같지만 선택epoch는 조건별로 달라질 수 있다. 총60조건, 최대120fit이다.

| 조건 | 입력 |
|---|---|
| A_conditions | 목표 in vivo 시간·용량·단위·등급 |
| B_conditions_vitro | A + 여러 시간·용량의 in vitro 발현·조건·관측mask |
| C_conditions_morgan | A + Morgan/presence |
| D_conditions_vitro_morgan | B + Morgan/presence |

소스 입력을 사용하지 않는 A/C는 학습 및 평가에서 **발현·출발조건·관측mask 전체를0**으로 한다. 목표조건과 화학구조는 그대로 둔다. 소스를 사용하지 않는 경우에도 공통 전처리는 학습에서만 적합하고 그 결과를 입력에서0으로 가린다. 모든 조건의 전체입력폭·명목파라미터수는 같으며 target 인코딩은 동일하다. 유효 입력/활성parameter 수는 다르므로 실질 모델용량까지 동일하다고 주장하지 않는다. B−A와D−C는 발현만의 효과가 아니라 출발조건/결측패턴을 포함한 in vitro 정보 묶음의 효과다.

구조는 기존Morgan을 사용한다. 이번에는 새로운2D선별/구조encoder/gating/VAE/diffusion을 추가하지 않는다. 작은은닉층의 기존MLP를 사용하지만 파라미터수는 실제model_dimensions.csv로 보고한다. 기존 단위미통일·인코딩·slot 문제도 수정하지 않았으므로 그대로 한계에 포함한다. 이 실험은 RAID재현이 아니다.

## 실행과 저장

`python scripts/compare_information_inputs.py --prepared /PREPARED --structure /MORGAN.csv --smiles /SAVED_SMILES.csv --out /NEW_OUTPUT --execute`

--execute 없으면 계획만 표시한다. source absent 반복을 완료한 뒤 present를 수행한다. 하위공통실행기가 저장SMILES에서기술자를계산하지만 이번MLP입력에는Morgan만 사용한다. PubChem검색·Morgan재생성·원자료다운로드는 하지 않는다. 원래 결과와 합치지 않고 새폴더에 새로 계산한다. 같은경로/코드/환경/옵션으로 재개할 때 완료조건을검증해재사용하며 불일치는거부한다. soft예산은 source모드별 적용이며 수행중fit을강제종료하지않는다. 부분결과를전체완료로보이지않는다.

화합물분할/targetID/기준선이두모드에서동일하고 네조건의명목모델크기가같은지 검사한다. 전체OOF평가를각모델seed에서먼저계산하고 평균/표준편차를보존한다. 기준선은한번만보고한다.

- summary.csv: 조건별RMSE/MAE/Pearson/Spearman/R²의mean/std
- classification_summary.csv: 임계값0.5/1.0 Macro-F1와증가/감소F1 등
- pooled_by_model_seed.csv, classification_by_model_seed.csv: 모델seed별전체OOF지표
- paired_rmse.csv: B−A, D−C, C−A, D−B (음수면앞조건오차감소)
- model_dimensions.csv: 입력차원·parameter수 기록

과거테스트를관찰한탐색실험이며5-fold를세초기화에서재사용한다. 독립분할3회/통계적유의성/독립확증시험으로표현하지않는다. 정보추가가개선되면해당입력/모델/분할의근거이지모든표현/생성모델의기여입증은아니다.

합성검사: 소스제거시목표조건보존/원래입력불변, 네조건전체OOF/분할/target/기준선/모델크기, 재개와변경설정거부. 실제서버학습재현과구분한다.
