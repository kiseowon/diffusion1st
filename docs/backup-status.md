# 서버 실행 백업의 확보·검증 상태

## 보존 위치

`results/controlled_latent_mlp_ddpm_20261104_044442_34d1be/`에 이번 비교의 원본을 보존했습니다.

| 경로 | 내용 |
| --- | --- |
| executed_cell.py, config.json, parent.json | 실제 비교 셀·설정·상위 실행 연결 |
| mlp/, ddpm/ | 각 모델의 checkpoint·예측·조건 지표·검증 이력·선택·완료 기록 |
| baseline.joblib, baseline_predictions.npz | 목표 조건 Ridge와 저장 예측 |
| inner/final_input_preprocess.joblib | 학습 단계별 입력 전처리 |
| inner/final_output_preprocess.joblib | 학습 단계별 잔차 PCA·척도 |
| dependencies/parent_decoder/ | source 변화량·기준선·입력을 준비한 상위 셀과 감사 기록 |
| dependencies/parent_lowrank/ | 최초 데이터·분할 준비 셀과 helper3개·분할·probe/구조/입력 hash 기록 |
| server_results.csv | 실제 서버 CSV 원본. 기존 results.csv는 채팅 출력 전사본으로 별도 유지 |
| backup_import_manifest.json | 원본 파일·서버 경로·보존 경로·hash 연결 |
| backup_checks.json | 수행한 검사와 미수행 범위 |

기존 단독 model.pt·predictions.npz는 추가 백업의 MLP 파일과 SHA-256이 일치해 mlp/ 파일로 확인했습니다. 초기 업로드 식별자는 artifact_manifest.json에 유지했습니다. 파일 내용을 고쳐 재구성한 원본이 아닙니다.

## 확인한 것

- 두 ZIP의 CRC 검사 통과, 원본 manifest의 총44파일 크기·SHA-256 일치.
- 서버 기록 helper_commit은 practice의3300adec5e204b6e4930b4cdf396d9fd0f197111. 첨부 helper3개가 상위 run.json의 hash와 일치.
- 상위 run.json 사본들의 바이트 일치,140개 고유 화합물 분할89/23/28, 지표의 테스트 화합물28개와 일치.
- 두 모델과 기준선의 예측 shape336×3453·유한값·동일 target ID 순서. 두 모델 지표 표의 ID와도 일치.
- 모델별 조건 MSE를 재집계한 RMSE가 complete.json 및 서버 보고와 일치. 선택 설정은 검증 이력의 최솟값과 일치.
- 실제 비교 셀의 구조 입력, x0-MSE, cosine30step·16sample, 입력/출력 훈련 전처리, 훈련 내 Ridge 잔차 구현을 읽어 확인.
- 보존 Python6개는 문법 파싱만 수행. 모델·joblib 역직렬화, 학습·추론은 실행하지 않음.

예측 ZIP에는 manifest 외에 동일한 code_and_models ZIP의 중복 사본도 들어 있었습니다. manifest 대상44개 파일을 보존했고 중복 ZIP을 추가로 Git에 저장하지 않았습니다.

## 아직 확인하지 않은 것

원자료 prepared_controls의 발현 NPY·조건/pair/probe 테이블, Morgan CSV 본문은 이 백업에 없습니다. 상위 run.json에 hash는 있으나 원자료를 받아 독립 재검사한 것은 아닙니다. 원자료 정답과 저장 예측의 직접 오차 재계산, checkpoint로 예측 재생성, 새 환경의 전체 학습 재현은 수행하지 않았습니다. 원래 실행 환경 전체 lock도 없으며 run.json에 기록된 버전은 상위 셀 실행 시 기록입니다.

구조 입력은 코드와 파일 경로로 확인했지만 PubChem 물질 동일성까지 검증한 것은 아닙니다. 주어진 구조 audit에는 LPS·TNFÉø·tunicamycin이 누락으로 기록돼 있습니다.

## 실행 의존성 주의

이 백업은 과거 실행 셀 보존이며 새 CLI 실행기를 작성한 것이 아닙니다. 비교 셀은 상위 decoder 셀의 전처리·Ridge·분할 객체를, 상위 decoder 셀은 저차원 셀의 데이터·probe 순서·분할·helper를 사용했습니다. helper 일부와 환경은 고정 practice 커밋에 의존합니다. 상위 decoder가 import하는 pathway_random_control.py 등 모든 외부 helper가 이번 ZIP에 별도 포함된 것은 아닙니다.

**셀을 Run All로 실행하면 상위의 불필요한 실험까지 다시 학습할 수 있습니다.** 즉시 재실행 안내로 사용하지 마세요. 별도의 복원 절차를 만들 때 원자료·고정 helper·전처리와 모델 구조를 연결하고, 코드 보존과 새 환경 재현 검증을 분리해야 합니다.
