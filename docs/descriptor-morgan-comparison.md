# MLP에서 Descriptor와 Morgan 비교

`scripts/compare_structure_representations.py`는 기본 / RDKit 2D Descriptor / 저장된 Morgan의 세 입력을 비교한다. AIVIVE 참고 PCA-free 직접 반응 MLP의 나머지 설정을 유지한다. pathway·Ridge·PCA는 사용하지 않는다. 기본은 계획 모드이고 `--execute`에서만 실행한다. 이 환경에서는 소분자로 만든 합성 데이터에 대한 검사만 수행했고 연구 데이터 학습은 하지 않았다.

## 필요한 파일

- 기존 prepared_controls의 발현 NPY·조건 CSV·pairs.csv.gz·probes.csv
- 기존 morgan_matched/morgan_binary.csv (compound + 이진 bit)
- 저장한 compound_pubchem_smiles.csv (필수 열 compound, smiles)
- Python 패키지 RDKit 및 기존 실행 환경의 numpy/pandas/sklearn/torch/joblib 등

SMILES를 PubChem에서 재검색하거나 Morgan을 재생성하지 않는다. 이름은 공백만 제거해 정확히 매칭하며 인코딩을 추측으로 수정하지 않는다. SMILES 이름 중복은 거부한다. SMILES의 물질 동일성이 검증됐다는 뜻은 아니다. 기존 Morgan의 생성 이력과 신규 Descriptor의 분자 처리 정책이 동일한지도 별개로 확인해야 한다.

## Descriptor 계산과 학습 전처리

RDKit `Descriptors.descList`에서 3D registry에 있는 이름을 제외한다. 이는 새 RDKit 2D 기술자 목록이며 RAID의 정확한 목록을 복구한 것은 아니다. 기술자 이름과 RDKit 버전을 각 run.json에 기록한다. 새로운 버전은 기술자 목록·계산값이 바뀔 수 있다. 이 환경에서 테스트한 설치 버전을 과거 연구 버전으로 표시하지 않는다.

저장된 SMILES 전체 분자에 대해 계산한다. conformer 생성·최대 fragment 선택·염 제거·입체구조 최적화는 하지 않는다. 파싱 실패나 개별 계산 예외/비유한 결과는 결측으로 기록한다. raw_2d_descriptors.csv와 descriptor_calculation_audit.csv에 원시 값·파싱 결과·계산 결측 수를 보존한다.

각 내부 학습과 외부 재학습 단계에서 따로 다음 절차를 적합한다.

1. 유효한 기술자가 있는 **고유 학습 화합물**만 사용한다. pair/target 수로 화합물의 비중을 늘리지 않는다.
2. 학습 화합물 중 결측/inf가 하나라도 있는 기술자 열을 제거한다. 분자 자체가 없거나 파싱이 안 된 화합물은 이 열 선별에 포함하지 않는다.
3. 학습에서 상수인 열을 제거한다.
4. 이름순으로 기술자를 보며, 앞서 남긴 기술자와 Pearson 절댓값이 **0.95 초과**면 해당 열을 제거한다. 상관된 두 열을 모두 지우는 방식이 아니다. 결정적인 탐욕적 규칙이며 예측력으로 선택하지 않는다.
5. 남긴 열에 학습 고유 화합물의 StandardScaler를 적합한다.
6. 검증·시험에서는 선별과 scaler를 그대로 사용한다. 새로 결측이 생긴 선택 기술자는 **학습 평균**으로 채워 표준화 후 0으로 만든다. 분자 전체가 없으면 전부 0 + presence 0이다. 분자는 있지만 일부 기술자만 결측이면 해당 값만 대치하고 presence는 1이다.

선별 목록·제거 이유·scaler·적합한 학습 화합물은 inner_preprocess.joblib/final_preprocess.joblib에 보존한다. held-out 분포를 보고 열을 고르지 않는다. 검사 후 유효 학습 구조가 3개 미만이거나 남는 기술자가 없으면 중단한다.

## 공정한 비교의 범위

기본/Descriptor/Morgan 모두 같은 모든 화합물·조건·분할·모델 seed를 사용한다. 구조 누락 화합물을 임의로 제외하지 않는다. structure_coverage.csv에서 Descriptor와 Morgan의 실제 이름 매칭 범위를 함께 확인한다. 범위가 다르면 성능 차이는 표현 외에 입력 가용성 차이도 포함한다.

구조 입력 폭은 max(Morgan bit+presence 수, 전체 Descriptor 수+1)로 고정한다. Descriptor는 선택 열을 앞에서부터 놓고 나머지는 0, Morgan은 원래 bit 순서로 놓고, presence는 마지막 열에 놓는다. 기본은 모두 0이다. 세 조건은 동일한 네트워크·명목 parameter 수를 사용한다. 활성 변수 수·정보량·최적 규제 강도가 같다는 뜻은 아니다.

발현 입력은 기존과 같은 train-fit MinMax source 반응과 source별 특징의 target별 평균이다. 조건은 dose log1p/시간 표준화/단위·등급 onehot, 정답은 train-fit probe별 MinMax 반응이다. 모든 probe를 출력하고 역변환 후 평가한다. MLP context256/head64, LeakyReLU0.2, Adam lr1e-5, MAE, batch64, max100epoch, 내부 검증1/25/50/75/100의 response RMSE로 선택한다. 구조 표현만 바꾸며 동일한 초기 예산을 사용한다. 하이퍼파라미터 최적성이 검증된 비교는 아니다.

5fold × model seed3 × 3입력 = 45조건, 최대90 fit. 분할 seed42는 기존에 관찰한 탐색 분할이다. 화합물5fold를 세 번 새로 나눈 것이 아니며 seed 반복만으로 독립 화합물 수가 늘지 않는다.

## 결과 파일

각 arm/basic, descriptor, morgan 폴더에 기존 반복 실행기 형식의 예측·모델·분할·검증·입력과 코드 hash를 저장한다. 원시 Descriptor는 열 선별 전 값이므로 CSV의 결측 자체가 오류는 아니다. run.json의 RDKit 버전/목록/SMILES hash/코드/입력 계약이 같아야 재개한다. 기존 소스 파일이나 입력을 바꾼 폴더에 덮어 실행하지 않는다.

- representation_summary.csv: 모델과 기준선의 pooled RMSE
- representation_by_seed.csv: 모델 seed별 OOF RMSE
- representation_paired_differences.csv: Descriptor−기본, Morgan−기본, Descriptor−Morgan (음수가 앞 방법의 낮은 오차)
- 각 arm은 zero response와 학습 시간×등급 평균 기준선도 포함

추가 지표는 기존 evaluate_repeated_predictions.py로 각 arm에 대해 계산한다. F1 임계값은 탐색적 반응 크기 기준이며 DEG 유의성 판정이 아니다. 실험군 간 공정한 비교는 동일 평가 화합물·목표 조건을 유지해야 한다. RAID와의 수치 직접 비교나 특정 모델 계열의 보편적 우월성을 의미하지 않는다.

```bash
python scripts/compare_structure_representations.py \
 --prepared /home/seowon1211/data1/conditioned_expression/prepared_controls \
 --structure /home/seowon1211/data1/conditioned_expression/morgan_matched/morgan_binary.csv \
 --smiles /YOUR/SAVED/compound_pubchem_smiles.csv \
 --out /home/seowon1211/data1/conditioned_expression/NEW_REPRESENTATION_RUN
```

기본은 계획만 출력한다. 실제 학습은 `--execute`를 추가한다. RDKit이 없으면 같은 Python 환경에 `python -m pip install rdkit`을 먼저 실행하고 커널을 재시작한다. 이는 새 비교용 환경 설치이며 과거 RDKit 버전 복원이 아니다.
