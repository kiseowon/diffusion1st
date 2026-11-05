# 시간별 입력 보존 모델의 반복 검증

실행기: `scripts/repeat_time_structure_pilot.py`. 단일 holdout 파일럿은 보존하고 별도 출력 폴더를 사용한다. 기존 `meeting_time_structure_pilot.py`에 fold/분할 seed/모델 seed/arm/family 선택을 추가했으며 이전 기본 실행은 여전히9조건이다.

## 승인한 반복 범위

- 구조 없음/basic, Morgan/morgan × MLP/cGAN/x0-DDPM
- 화합물5fold, 분할 seed42. 화합물당 한 번씩 외부 테스트.
- 모델 초기화 seed42/43/44. 새로운5fold를 세 번 만든 것이 아니다.
- 총90조건, 내부 학습 후 외부 재학습까지 최대180 fit.
- Descriptor는 앞선 서버 파일럿에서 내부 검증RMSE24~30의 이상 신호가 있었으므로 이 반복에서는 제외한다. 원인 확정·수정 완료를 주장하지 않는다.

시간별 입력 슬롯·관측mask·source PCA32/target PCA32·context256/hidden64·Adam lr1e-4·batch128·DDPM20step/sample4 등은 [파일럿](meeting-time-structure-pilot.md)과 같다. 최대epoch만100으로 늘린다. 내부 검증은1/5/10/15/20/40/60/80/100이며 각 모델/seed/arm/fold에서 원래 반응공간RMSE로 선택한다. PCA/스케일러는 내부 학습에만 적합하고 최종 외부 학습 전체에 새로 적합한다. 구조 없는 arm은 동일한 구조입력 슬롯을0으로 채워 같은 계열끼리 parameter 수를 맞춘다.

학습률·PCA차원·epoch상한·배치 등은 초기값이고 최적성 미검증이다. AIVIVE 참고 설정과 자체 조정은 파일럿 문서에서 구분한다. 테스트를 반복 관찰하므로 확증 시험이 아니다. 전 시간대source를 쓰며 미래 시간 forecasting을 주장하지 않는다. 이번 비교에는 pathway가 없다.

## 시간 제한과 재개

기본 전역60분은 fold/seed 작업 단위 사이에서 확인한다. 개별 fold/seed에는15분의 소프트 예산을 두어 다음 모델 작업의 시작을 제한한다. 진행 중인 모델은 끝내므로 **60분 또는75분 이내 종료를 보장하지 않는다.** hash·전처리·평가 시간도 필요하다. 별도 예산을 알고 조정할 수 있으나 기존 폴더는 설정 계약이 같아야 재개한다.

모델 seed별로5fold를 먼저 완료하는 순서다. 예산으로 중단돼도 완전히5fold를 마친 seed의 OOF 결과는 집계할 수 있다. 일부fold만 완료한 seed는 전체 OOF 집계에서 제외하고 by_fold_model_seed.csv에만 남긴다. evaluation_status.json에 전체fold를 마친 seed와 예정 seed를 기록하므로 부분 결과를 전체 완료로 제시하지 않는다.

같은 명령·출력 폴더로 다시 실행하면 완료 marker·input/source/version 계약·artifact hash를 검사하고 완료한 학습은 건너뛴다. 입력/코드를 바꾼 뒤 예전 폴더를 재개하지 않는다. `--execute` 없으면 입력을 읽거나 출력을 만들지 않는 계획 모드다.

## 집계

`pooled_by_model_seed.csv`는 각 모델seed에서5fold의 모든 예측을 합쳐 실제target×probe오차로 계산한다. fold별RMSE의 단순 평균이 아니다. 각target은 fold간 한 번씩만 등장하는지 검사한다.

- summary.csv: 전체OOF의 RMSE/MAE/Pearson/Spearman/R2를 모델seed에 걸쳐 평균·표준편차
- classification_summary.csv: 임계값0.5/1의 F1·방향 지표.0~1이며 %표시는100배
- paired_rmse_differences.csv: 같은 모델seed의 Morgan−basic와 DDPM−MLP/cGAN.음수가 앞 방법의 낮은RMSE
- oof_condition_metrics.csv: 화합물/target별MSE·MAE
- 기준선은 같은 외부학습의 시간×등급평균과zero response. 모델seed마다 중복 집계하지 않고 같은fold의 기준선이 일치하는지 검사
- repeat_status.json: 전체 완료 여부·예산 중단 여부

표준편차는 같은fold 배치에서 모델초기화 변동만 나타낸다. 화합물sampling CI·여러 분할 반복의 변동·통계검정이 아니다. 예측ensemble도 아니다. 분류임계값은 DEG 통계적 유의성이나 간독성 판정이 아니다. 독성/pathway결과는 이번 실험 범위에 없다.

실제 연구 데이터 학습은 사용자 서버에서 실행한다. cloud에서 합성 데이터의 fold/seed별실행·정답누수방지·중복평가방지·완료재개를 검사한 것과 구분한다.

```bash
python scripts/repeat_time_structure_pilot.py \
 --prepared /home/seowon1211/data1/conditioned_expression/prepared_controls \
 --structure /home/seowon1211/data1/conditioned_expression/morgan_matched/morgan_binary.csv \
 --smiles /home/seowon1211/data1/compound_pubchem_smiles.csv \
 --out /home/seowon1211/data1/conditioned_expression/NEW_REPEATED_TIME_RUN
```

`--execute`를 붙이면 실행한다. source snapshot과 명령을 결과 옆에 보존한다. 처음 파일럿과 최대epoch가 달라졌으므로 최종성능 차이를 반복만의 효과로 해석하지 않는다.
