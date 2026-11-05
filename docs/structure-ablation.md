# 동일 설정의 구조 정보 on/off 비교

현재 세 모델 비교는 모두 Morgan을 사용했습니다. 구조 입력의 효과를 검증하려면 같은 조건에서 구조 정보를 주지 않는 대조가 필요합니다. `repeated_latent_validation.py --no-structure`로 구현했습니다.

## 바꾸는 것과 유지하는 것

Morgan의 모든 binary bit와 presence flag를 함께0으로 만듭니다. presence만 남겨 구조 누락 패턴을 입력하는 것은 아닙니다. source 변화량·실험 조건·입력 PCA·잔차 PCA·공통 Ridge·loss·학습 예산·생성 수는 유지합니다.

원래 구조 입력 자리는 남기므로 같은 fold에서 입력 차원·parameter 수·초기 가중치가 같습니다. 유효 정보량이나 유효 모델 용량까지 같다는 뜻은 아닙니다. 구조 파일은 off에서도 같은 열 수를 유지하고 입력 hash·availability를 확인하기 위해 필요합니다.140개 화합물을 유지하며 구조 누락3개를 on에서만 제외하지 않습니다.

## 완료된 on 결과 재사용

기존 on 결과는 `three_family_repeated_20261104_074715_159844`입니다. 원래9a71e16 runner를 고정 SHA-256으로 식별하고 새 off 구현의 구조 flag·0 입력·감사 기록 추가를 코드로 검토했습니다. 알 수 없는 임의 source를 같은 실험이라고 허용하지 않습니다.

```bash
python scripts/repeated_latent_validation.py \
  --prepared /home/seowon1211/data1/conditioned_expression/prepared_controls \
  --structure /home/seowon1211/data1/conditioned_expression/morgan_matched/morgan_binary.csv \
  --out /새로운/structure_off_폴더 \
  --families mlp cgan ddpm \
  --no-structure \
  --paired-on-run /home/seowon1211/data1/conditioned_expression/three_family_repeated_20261104_074715_159844
```

기본은 계획만 출력합니다. `--execute`로 off 실제 학습을 시작합니다.5-fold×3seed×3family의45조건·최대90회 신경망 fit이며 기존 on은 다시 학습하지 않습니다. 별도로 진행 중인 튜닝 최선값을 off에만 적용하지 않습니다. 기존 on과 같은 고정 설정으로 비교합니다.

`--paired-on-run`은 학습 전에 on의 완료 상태, 입력 hash, 구조 외 protocol, 분할·모델 seed, fold 수, family, thread 수, 패키지 버전과 알려진 source를 확인합니다. 다르면 학습 전 중단합니다. 새 계약에 on의 run.json hash도 기록합니다. 과거 on의 재개에는 과거 고정 코드를 사용하고 새 off는 별도 폴더에 저장합니다.

## 완료 후 대응 비교

```bash
python scripts/compare_structure_ablation.py \
  --on-run /완료된/on_run \
  --off-run /완료된/off_run \
  --out /새로운/비교폴더
```

모델을 로드하거나 학습하지 않고 유효 완료 marker의 조건별 MSE를 비교합니다. 입력·protocol·버전을 다시 대조하고, 저장 compound split의 바이트 일치와 Ridge/시간×등급 baseline 예측·target 순서 일치를 확인합니다. 모델 산출물 hash도 검사하고 target metadata를 일대일로 연결합니다.

- paired_by_split_model_seed.csv: 모델·seed별 전체OOF RMSE on/off와 대응 차이·개선율.
- summary.csv: 모델별 seed 평균과 대응 차이. **on−off가 음수면 구조 on의 오차가 낮습니다.**
- paired_condition_metrics.csv: 같은 target의 대응 MSE.
- compound_bootstrap.csv: seed/repeat MSE를 먼저 평균하고 화합물을 재표집한 고정 예측 구간. 재학습 불확실성 전체를 나타내지 않습니다.
- comparison.json: 비교 대상·source 호환성·검사 hash 기록.

비교 summary의 RMSE는 unit RMSE의 평균입니다. 원래 runner의 summary(MSE 평균 후 제곱근)와 끝자리가 조금 다를 수 있고, 예측 ensemble이 아닙니다. 추가 F1·방향 평가에는 off의 저장 예측에 기존 evaluate_repeated_predictions.py를 적용할 수 있습니다.

## 해석과 검사 범위

이것은 **현재 Morgan+presence 표현을 추가한 효과**의 탐색 ablation입니다. 무작위 fingerprint보다 화학적 구조 특이성이 유리하다고 검증한 비교는 아니고 scaffold 외삽도 아닙니다. 세 모델 seed는 같은 화합물 fold를 공유하며, 이미 관찰한140개 화합물이므로 새로운 독립 확증 시험은 아닙니다.

off가 같거나 더 좋은 결과도 그대로 받아들이고, on이 좋아지도록 설정을 계속 고르지 않습니다. 구조 효과와 DDPM의 다른 모델 대비 효과는 별개의 대응 비교입니다.

합성 검사에서는 off 계획의 무실행, Morgan/presence 외 입력 불변, 학습 잠재 정답 불변, 세 계열의 초기 가중치·parameter 수 일치, 축소한 합성 on/off 실행과 비교, 설정 불일치 거부를 확인했습니다. 코드 작성 환경에서는 연구 데이터 off 학습을 실행하지 않았습니다.
