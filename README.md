# IVIVE: 저차원 잔차 예측 탐색 실험

구조 정보와 in vitro 반응을 이용해 미관측 화합물의 조건 평균 in vivo 반응을 예측합니다. 이 저장소는 기존 [practice](https://github.com/kiseowon/practice) 연구와 분리한 새 탐색 기록입니다.

현재 보존한 비교는 **동일한 입력·출력 PCA·Ridge 기준선 아래에서 잠재 MLP와 조건부 x₀-DDPM의 잔차 보정을 비교한 실험**입니다. 학습·평가 대상은 개별 동물 분포가 아닌 처리군−대조군 조건 평균 반응입니다.

| 방법 | 보고된 테스트 반응 RMSE ↓ |
| --- | ---: |
| 목표 조건 Ridge | 0.312839 |
| Ridge + 잠재 MLP | 0.311859 |
| Ridge + 잠재 x₀-DDPM | 0.310624 |

DDPM의 MLP 대비 RMSE 차이는 −0.001235입니다. **단일 화합물 분할·모델 seed42의 탐색 결과**이며, 모든 모델이 구조 입력을 사용하므로 구조의 효과를 검증한 표는 아닙니다. 같은 분할에서 별도로 보고된 시간×용량 등급 평균 기준선 0.310385보다 낮지는 않았습니다.

- [실험 설계·전처리·학습·해석의 한계](docs/controlled-latent-comparison.md)
- [보고된 결과 CSV](results/controlled_latent_mlp_ddpm_20261104_044442_34d1be/results.csv)
- [출처·검증 상태](results/controlled_latent_mlp_ddpm_20261104_044442_34d1be/provenance.json)
- [제공한 실행 셀의 설정 기록](results/controlled_latent_mlp_ddpm_20261104_044442_34d1be/protocol.json)

수치 요약은 사용자가 붙여 준 서버 출력에서 전사했습니다. 이후 첨부된 두 모델의 조건별 MSE를 재집계해 RMSE가 완료 기록과 일치함을 확인했고, 검증 이력의 최솟값과 선택 설정도 일치했습니다. 이는 원자료와 예측을 직접 비교한 지표 재계산이나 학습 재현이 아닙니다. [첨부 원본 manifest](results/controlled_latent_mlp_ddpm_20261104_044442_34d1be/artifact_manifest.json)와 [검사 기록](results/controlled_latent_mlp_ddpm_20261104_044442_34d1be/artifact_checks.json)을 보존했습니다.

예측 NPZ 하나와 checkpoint 하나도 첨부됐지만 어느 모델의 파일인지 확인되지 않아 로컬 `unassigned/`에 구분했습니다. 자동 승인 검토가 이 두 binary 파일의 외부 전송을 거부하여 Git에서는 제외했으며 GitHub에 업로드하지 않았습니다. 파일별 hash와 로컬 검사 결과만 기록했습니다. 체크포인트는 역직렬화하거나 실행하지 않았습니다. 서버의 executed_cell.py, 전처리 객체, 정확한 분할·입력 hash 원본은 아직 없습니다. **현재 저장소는 실행 가능한 전체 백업이 아니라 설계·일부 산출물·검증 상태의 기록입니다.**
