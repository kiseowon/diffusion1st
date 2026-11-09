# 입력 PCA64 대 전체 probe 반복 비교

목적은 같은 화합물 분할·모델 seed·구조 표현·손실·학습/epoch선택 설정에서 입력 파이프라인을 비교하는 것이다. 기본은 MLP, 구조 없음/Morgan, 화합물5-fold, 모델seed42/43/44, 두 입력 모드다. 총60조건·최대120fit이며 실제 학습은 사용자 서버에서만 수행한다. DDPM/cGAN은 `--families`로 명시해야 포함된다. 입력 한 조건의 probe 특징만 PCA64 또는 전체3453으로 바꾸고 시간·용량별 slot, 목표 조건과 출력전체probe는 유지한다. 두 모드를 새 폴더에서 다시 계산하며 과거 결과와 합치지 않는다.

## 통제한 것과 달라지는 것

구조 슬롯은 separate 레이아웃으로 고정하고 같은 입력모드의 basic/Morgan 입력폭은 같다. 시간/용량조건, 화합물 분할과 초기화seed, context256/head64, max100epoch/내부검증epoch선택, 최종재학습과 전체OOF지표집계를 맞춘다. scaler/PCA는 내부학습 또는 최종학습 화합물에서만 적합한다. fold별 같은화합물/target 및 기준선 예측이 일치하는지 검사한다.

**첫 층 parameter 수는 입력 차원에 따라 다르다.** PCA 입력은 학습MinMax→PCA→scoreStandardScaler, 전체 입력은 학습MinMax다. 따라서 이 비교는 두 전처리/입력 파이프라인의 성능 비교이지 동일 모델용량·스케일에서 PCA 정보손실만 분리한 인과 실험이 아니다. model_dimensions.csv에 실제 입력/출력차원·parameter·대략적인메모리규모를 기록한다. 고정slot 설계의 미관측조건 문제는 이 실험에서 수정하지 않았으므로 동일한 기존 한계를 갖는다.

## 실행

`python scripts/compare_input_pca_full.py --prepared /PREPARED --structure /MORGAN.csv --smiles /SAVED_SMILES.csv --out /NEW_OUTPUT --execute`

기본은 계획 모드이고 --execute로만 학습한다. pca모드 반복을 마친 뒤 full모드 반복을 수행한다. full입력은 모델이 커지고 CPU 계산이 오래 걸릴 수 있다. 소프트예산은 모드별로 적용하며 수행중fit은 중단하지 않는다. 부분 완료는 비교완료로 표시하지 않는다. 같은 코드·환경·입력·경로·설정으로 재개하면 완료된 조건은 hash/계약을 확인한 뒤 재사용한다.

## 보고

전체fold 예측을 합쳐 모델seed별 지표를 계산하고 seed간 평균/표준편차를 보고한다. paired_by_model_seed.csv의 full_minus_pca_rmse가 음수면 전체입력이 좋다. RMSE가1차, MAE/Pearson/F1가보조지표다. 모든seed/구조조건/기준선을 보존하고 최저실행만 선택하지 않는다. pooled_metrics.csv, comparison_summary.csv, combined_classification_summary.csv도 보존한다. 5-fold가 세seed에서 공유되므로 독립적인분할3회가 아니고, p-value/화합물bootstrap CI는 제공하지 않는다. 과거시험결과 관찰이 있는 탐색평가이며 독립확증시험으로 표현하지 않는다.

이 환경의 합성검사는 두모드전체OOF, 같은target/기준선/분할, 파라미터기록, 재개, 설정불일치거부만 확인한다. 실제 실험재현 완료와 구분한다.

## 구조 3종 × 모델 3종의 전체 비교

`--arms basic morgan descriptor --families mlp cgan ddpm`으로 구조 없음/Morgan/2D와 세 모델을 모두 명시할 수 있다. 두 입력 방식 × 3구조 × 3모델 × 5-fold × 3초기화 = **270조건, 최대 540fit**이다. 과거 일부 결과와 합치지 않고 동일 코드로 두 입력 방식을 함께 계산한다. `both`도 선택 가능하지만 이 계획에는 포함하지 않는다. 기본 CLI 범위는 기존 MLP·구조 없음/Morgan을 유지한다.

서버용 독립 실행 셀은 [launchers/run_complete_pca_full.py](../launchers/run_complete_pca_full.py)다. 입력 경로를 먼저 검사하고 저장소 snapshot을 새 결과 폴더에 복제한다. 재개할 때는 `RESUME_ROOT`를 해당 폴더로 지정하며 코드를 새로 받지 않는다. 모드별 소프트 예산은 7일로 설정했지만 실제 소요 시간 추정은 아니다. CPU 전체 입력 학습은 큰 메모리와 긴 시간이 필요할 수 있다. 임의로 실패 seed를 빼거나 PCA로 대체하지 않고 부분 완료로 남긴다.

합성 데이터 검사는 세 구조·세 모델의 두 파이프라인, 대응 집계, 재개 및 설정 변경 거부를 확인한다. 실제 270조건 학습 결과 확보와는 구분한다.
