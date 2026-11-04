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
