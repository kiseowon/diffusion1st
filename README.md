# IVIVE: 저차원 잔차 예측 탐색 실험

구조 정보와 in vitro 반응을 이용해 미관측 화합물의 조건 평균 in vivo 반응을 예측합니다. 이 저장소는 기존 [practice](https://github.com/kiseowon/practice) 연구와 분리한 새 탐색 기록입니다.

현재 보존한 비교는 **동일한 입력·출력 PCA·Ridge 기준선 아래에서 잠재 MLP와 조건부 x₀-DDPM의 잔차 보정을 비교한 실험**입니다. 학습·평가 대상은 개별 동물 분포가 아닌 처리군−대조군 조건 평균 반응입니다.

| 방법 | 보고된 테스트 반응 RMSE ↓ |
| --- | ---: |
| 목표 조건 Ridge | 0.312839 |
| Ridge + 잠재 MLP | 0.311859 |
| Ridge + 잠재 x₀-DDPM | 0.310624 |

DDPM의 MLP 대비 RMSE 차이는 −0.001235입니다. **단일 화합물 분할·모델 seed42의 탐색 결과**이며, 두 보정 모델이 모두 구조 입력을 사용하므로 구조의 효과를 검증한 표는 아닙니다. 같은 분할에서 별도로 보고된 시간×용량 등급 평균 기준선 0.310385보다 낮지는 않았습니다.

- [실험 설계·전처리·학습·해석의 한계](docs/controlled-latent-comparison.md)
- [보고된 결과 CSV](results/controlled_latent_mlp_ddpm_20261104_044442_34d1be/results.csv)
- [출처·검증 상태](results/controlled_latent_mlp_ddpm_20261104_044442_34d1be/provenance.json)
- [제공한 실행 셀의 설정 기록](results/controlled_latent_mlp_ddpm_20261104_044442_34d1be/protocol.json)

수치 요약은 사용자가 붙여 준 서버 출력에서 전사했습니다. 이후 첨부된 두 모델의 조건별 MSE를 재집계해 RMSE가 완료 기록과 일치함을 확인했고, 검증 이력의 최솟값과 선택 설정도 일치했습니다. 이는 원자료와 예측을 직접 비교한 지표 재계산이나 학습 재현이 아닙니다. [첨부 원본 manifest](results/controlled_latent_mlp_ddpm_20261104_044442_34d1be/artifact_manifest.json)와 [검사 기록](results/controlled_latent_mlp_ddpm_20261104_044442_34d1be/artifact_checks.json)을 보존했습니다.

폴더 구조를 유지한 추가 백업에서 실제 실행 셀·상위 실행 셀·분할·전처리·Ridge·두 모델의 checkpoint와 예측을 확보했습니다. [백업 검사](results/controlled_latent_mlp_ddpm_20261104_044442_34d1be/backup_checks.json)에서 manifest의44개 파일 hash/크기 일치와 helper3개의 서버 기록 hash 일치를 확인했습니다. 이전 unassigned 두 파일도 **MLP 원본과 SHA-256이 같아** mlp/로 정리했습니다. 첨부 원본은 변경하지 않았고, 서버 결과 CSV는 server_results.csv로 따로 보존했습니다.

- [서버 실제 비교 셀](results/controlled_latent_mlp_ddpm_20261104_044442_34d1be/executed_cell.py)
- [상위 셀·복원 의존성·현재 검증 범위](docs/backup-status.md)

checkpoint와 joblib은 역직렬화하거나 실행하지 않았습니다. **원자료 발현·조건/pair/probe·Morgan CSV와 새 환경의 전체 재실행 검증은 아직 없습니다.** 파일 확보·hash 검사와 학습/추론 재현은 구분합니다. 보존 셀들은 상위 커널 변수에 의존하므로 독립 실행용 프로그램이 아닙니다.

## 고정 설정 반복 평가 실행기

[반복 검증 설계·실행 안내](docs/repeated-latent-validation.md)와 [실행 코드](scripts/repeated_latent_validation.py)를 추가했습니다. 기본은 화합물5-fold × 모델seed3개 × MLP/DDPM으로30조건·최대60회 신경망fit이며, 계획만 출력합니다. `--execute`를 붙여야 실제 학습합니다. Ridge·시간×등급 평균·반응0 기준선을 같은 target으로 비교합니다. 코드 작성 환경에서는 작은 합성 검사만 했고 연구 데이터 반복 학습은 실행하지 않았습니다.

[저장 예측의 추가 평가](docs/additional-evaluation.md): 재학습 없이 MAE·R²·target별 Pearson/Spearman·방향 정확도·3분류 F1을 기준선과 함께 계산합니다. 진폭 threshold 기반 사후 탐색 지표이며 통계적 차등발현 판정이 아닙니다.

## 보고된 추가 평가 결과

[5-fold·모델seed3개의 추가 지표](results/additional_metrics_20261104_071610_11d5d7/README.md)를 보존했습니다. 서버 출력의6자리 전사본이며 원본 평가CSV는 아직 미확보입니다. DDPM은 목표 조건 Ridge보다 개선됐지만 MLP보다 일관되게 좋지는 않았고, 시간×등급 평균이 RMSE·MAE·R²에서 가장 좋았습니다. 94~98% 분류accuracy는 변화작음 다수 class의 영향을 크게 받으며, 큰 반응의 방향accuracy와 up/down F1을 구분해야 합니다.

## 세 모델 계열의 공통 조건 비교

[기준선의 의미와 예시](docs/baselines.md)와 [MLP·cGAN·DDPM 비교 설계](docs/three-family-comparison.md)를 추가했습니다. `--families mlp cgan ddpm`은45조건·최대90회신경망fit을계획합니다. 기본은기존MLP/DDPM이고실제학습은별도`--execute`가필요합니다. 이번확장에서는연구데이터학습을실행하지않았습니다.

## 발현 입력 출처 확인

[발현 출처와 전처리](docs/expression-source.md): 사용자 서버의 vitro_train_test.csv·vivo_train_test.csv가 AIVIVE Zenodo14984579 배포 파일의 MD5와 각각 정확히 일치했습니다. 논문·README의 RMA 정규화 설명과 연결해 AIVIVE가 전처리한 배포 발현 자료를 사용한 것으로 기록합니다. 이는 사용자 서버의 checksum 검사이며 이 환경에서 원시 CSV 다운로드·RMA 재실행·추가 학습을 수행한 것은 아닙니다.

## DDPM 설정의 이유와 민감도 계획

[설정 이유·검증 범위](docs/ddpm-hyperparameter-rationale.md)와 [내부 검증 전용 실행기](scripts/validate_ddpm_architecture.py)를 추가했습니다. context/hidden 폭을 한 번에 하나씩 바꾸는5후보×3seed의15회 내부 fit 계획이며, 기본은 계획 모드입니다. 외부 테스트 평가·최종 refit은 포함하지 않고 연구 자료에서는 아직 실행하지 않았습니다.

[DDPM 조합 탐색](docs/ddpm-joint-tuning.md): hidden×학습률×weight decay의12조합을 내부 검증하고 상위2개와 reference에 추가 초기화2개를 적용하는 실행기를 추가했습니다. 기본은 계획 모드,최대18회 내부fit이며 외부 test 평가와 최종refit은 하지 않습니다.

## 구조 정보 대조

[동일 설정 구조 on/off 비교](docs/structure-ablation.md)를 추가했습니다. `--no-structure`는 Morgan과 presence만0으로 바꾸고 입력 차원과 나머지 설정을 유지합니다. `--paired-on-run`으로 학습 전에 기존 on의 입력·설정·환경을 확인하고, 완료 후 비교 코드가 분할·기준선·대응 target을 검사합니다. 실제 연구 off 학습은 이 작업에서 실행하지 않았습니다.

## 구조 대조의 보고 결과

[구조 on/off 대조 결과](results/structure_ablation_20261105_020903_dfd7fc/README.md)를 보존했습니다. 고정 설정에서 구조 on이 평균RMSE를 MLP·DDPM에서약0.27%,cGAN에서약0.16% 낮췄습니다. MLP는3개seed 모두,DDPM/cGAN은2개seed에서 낮았습니다. 이는 단일분할의 작은 탐색적 개선 신호이며 모든지표의 개선이나 일반적 구조효과 입증은 아닙니다. 서버원본CSV가 아닌 첨부로그와 표시표 전사본으로 구분했습니다.

[코드 학습·연구미팅 PPT 초안](docs/presentation-and-code-study.md)을 추가했습니다. 기본 모델과 현재 파라미터 탐색을 구분하고, 슬라이드 문장·도식·발표 노트·함수별 공부 순서를 담았습니다. 진행 중인 조합 탐색은 완료 결과로 표시하지 않았습니다.

- [네 입력 × MLP/cGAN/DDPM 직접 반응 예측 실행 안내](docs/factorial-response-validation.md): 기본/구조/pathway/둘 다, Ridge 없는 별도 탐색 프로토콜. 기본은 계획 모드이며 실제 학습 결과는 아직 없음.

- [AIVIVE를 참고한 PCA 없는 직접 반응 예측](docs/aivive-inspired-no-pca.md): 구조 없음/있음 × MLP/cGAN/DDPM. 참고한 설정과 변경한 설정을 구분한 별도 탐색 코드이며 AIVIVE 원본 재현이 아님.

- [MLP의 Descriptor/Morgan 비교](docs/descriptor-morgan-comparison.md): 기본/2D Descriptor/Morgan, train-only 변수 선별·표준화와 동일 parameter 수를 적용한 별도 실행기. RAID 원본 재현이 아니며 실제 연구 데이터 결과는 아직 없음.
