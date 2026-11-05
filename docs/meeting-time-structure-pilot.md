# 첫 미팅용 시간 보존 파일럿

실행기 `scripts/meeting_time_structure_pilot.py`는 화합물 미관측 예측의 짧은 탐색 코드다. 입력 조건을 평균하던 이전 코드와 달리 시간·등급·단위별 슬롯과 관측 mask로 각각 보존한다. 구조 없음 / Morgan / RDKit 2D Descriptor × MLP/cGAN/x0-DDPM = 9조건을 실행한다. pathway나 지식그래프는 이번 범위에 없다.

## 시간·용량 입력

실제 source pair에 등장하는 in vitro 조건을 모두 사용한다. 각 `(time_h, level, unit)`를 슬롯으로 만들고, 같은 화합물에서 같은 key의 조건이 여러 개면 dose 오름차순으로 추가 슬롯을 둔다. 슬롯 schema는 **사용 가능한 source 메타데이터만**으로 미리 정하며 발현값이나 in vivo 정답으로 만들지 않는다. 이 방식은 입력 schema에 테스트 화합물의 조건 가용성 메타데이터도 포함한다. 발현 전처리와 모델은 학습 데이터로만 적합한다. 모든 슬롯의 시간/용량/등급/unit와 관측 여부를 입력하므로 시간별 반응이 평균되지 않는다. 빠진 슬롯은 모든 값0 + 관측mask0이며 실제 관측값0과 구분한다.

화합물별 전체 슬롯을 펼치고 목표 in vivo dose/time/unit/level을 추가한다. source encoder는 학습 고유 조건의 MinMax→PCA 최대32→score StandardScaler, source와 target 조건은 각각 학습 데이터의 log1p dose/시간 StandardScaler + unit/level OneHotEncoder이다. 서로 다른 단위를 물리적으로 변환하지 않는다. 각 화합물에서 관측된 여러 시간·용량을 쓰는 것으로 시간적 forecasting이나 하나의 농도에서 연속적인 trajectory를 배운다고 보장하지 않는다. 미래 in vitro 측정까지 포함할 수 있으므로 인과적 조기 예측으로 해석하지 않는다.

정답은 기존 prepared_controls의 실제 treated−matched control 반응이다. 처리군과 대응 대조군이 있어 정답을 계산할 수 있는 vivo target만 사용하고, 모든 모델·구조 조건은 같은 target을 평가한다. 테스트 in vivo 대조군은 정답 계산에만 쓰인다. target은 내부 학습 반응의 PCA 최대32 + 잠재score 공통 RMS로 변환하고 원래 probe 공간으로 복원해 평가한다. Ridge는 없다. PCA/scaler/Descriptor 선별은 내부 학습에만 적합하고 epoch 선택 후 외부 학습 전체에 다시 적합한다.

## 짧은 비교 예산과 설정 출처

화합물 seed42의 5fold 중 **fold0 하나만** 평가한다. 140개 기준 내부89/검증23/테스트28이며 모델 seed42 하나다. 5fold 전체 검증이나 seed 반복 완료로 표시하면 안 된다. 이미 사용한 화합물 결과를 재관찰하는 탐색 실험이다.

9조건, 최대18 fit. 최대20epoch, validation1/5/10/15/20, batch128, context256/head64, LeakyReLU0.2, DDPM20step·생성4회 평균. MLP는 PCA score MAE, cGAN은 10MAE+2BCE(no cycle), DDPM은 x0-MSE다. 원래 반응 공간 validation RMSE로 epoch를 선택하고 test에서 최종 평가한다. sample4는 계산 절충이며 생성 평균의 Monte Carlo 잡음이 남는다.

AIVIVE에서 참고한 것은 학습 MinMax, 재구성+적대적 손실의 형태와10:2, 판별기 SGD lr1e-4/momentum0.9다. 작은 은닉폭은 공개 판별기 폭256/64를 참고한 별도 적응 설계다. **예측모델 Adam lr1e-4는 AIVIVE의1e-5와 다르며 짧은 파일럿을 위한 초기값이다.** PCA차원32, batch128,20epoch, DDPM20step/sample4는 우리가 제한한 예산이다. dropout/cycle/원본 대형Generator는 없다. 최적성·수렴은 검증되지 않았으며 AIVIVE 재현이 아니다. 상한epoch 선택 시 학습 부족 가능성을 명시한다.

Descriptor는 저장된 SMILES의 2D registry에서 계산하며 PubChem 검색은 하지 않는다. 학습 고유 화합물에서 결측열/상수열/abs(corr)>0.95를 제거하고 표준화한다. 미관측 결측은 학습 평균, 구조 부재는 0+presence0. Morgan은 기존 binary+presence를 사용한다. 세 구조 arm은 같은 구조입력 폭으로 padding하여 명목 parameter 수를 맞춘다. 표적 compound의 구조 누락은 structure_coverage.csv에 보존한다. Descriptor와 Morgan의 분자 처리·가용성이 같다고 독립 검증한 것은 아니다. RAID descriptor 목록의 재현이 아니다.

## 시간 제한·저장·재개

기본 `--budget-minutes 60`은 **새 작업 시작에 대한 소프트 제한**이다. 현재 진행 중인 job을 강제로 중단하지 않으므로 총60분 이내 완료를 보장하지 않는다. 입력 로딩·hash·Descriptor 계산·전처리도 시간에 포함한다. 제한 도달 시 partial.json과 완료한 결과를 남긴다. 미완료 arm을 결과에서 조용히 제외해 최종9조건 결과로 표시하지 않는다. 이후 동일한 입력/소스/설정/출력경로로 재개하면 complete.json과 artifact hash가 유효한 job은 건너뛴다. budget을 포함한 설정을 바꾸면 새 폴더가 필요하다.

출력 results.csv에는 완료한 모델 및 zero/time-level 기준선의 RMSE/MAE/Pearson/Spearman/R2가 있다. classification.csv에는 탐색적 임계값0.5/1의 macroF1·증가/감소F1·방향정확도 등이 저장된다. F1은0~1이므로 발표용%는100을 곱한다. 기준선 시간×등급 평균은 외부 학습의 화합물별 평균 후 화합물 간 평균이다. 준비·학습·평가를 모두 수행한 결과는 서버 실행 후에만 주장한다.

source_slots.json, source control/target control audit, compound_split.csv, structure_coverage.csv, 전처리joblib, 모델checkpoint, predictions.npz, validation_history.csv, run.json의 입력/코드hash·버전을 보존한다. 현재 cloud에서는 합성 검사를 수행했으며 연구 자료 학습은 실행하지 않았다.

```bash
python scripts/meeting_time_structure_pilot.py \
 --prepared /home/seowon1211/data1/conditioned_expression/prepared_controls \
 --structure /home/seowon1211/data1/conditioned_expression/morgan_matched/morgan_binary.csv \
 --smiles /YOUR/SAVED/compound_pubchem_smiles.csv \
 --out /home/seowon1211/data1/conditioned_expression/NEW_MEETING_PILOT
```

기본은 입력을 읽지 않는 계획 모드다. `--execute`에서만 학습한다. 실제 완료 시간은 서버 자원에 따라 다르다. 다른 대형 학습과 동시에 실행하면 시간이 늘어날 수 있다.
