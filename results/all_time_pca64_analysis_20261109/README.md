# 전체시간 PCA64 실험 분석 근거

[분석 문서](../../docs/all-time-pca64-performance-analysis.md)를 참고하세요. 주 대상은134개 24h toy가 아니라140개 화합물의 전체시간·용량 실험입니다.

- full_repeated_server_log.txt: 사용자 서버의180조건 완료 원문.
- reported_jobs_and_epoch_selection.csv: START/COMPLETE와마지막best검증기록에서추출. test RMSE는6자리,검증dict값은로그의정밀도입니다.
- epoch_selection_summary.csv: 모델별선택epoch집계. 상한선택이수렴실패의입증은아닙니다.
- reported_model_dimensions.csv: 실제입력/파라미터수의보고값.
- reported_regression_summary.csv / reported_classification_summary.csv: 화면에표시된전체OOF요약. 원본CSV의전체정밀도는아닙니다.
- input_range_server_log.txt / fixed_slot_masking_server_log.txt / fixed_model_sampling_server_log.txt: 서로다른후속진단의원문. 새로운학습반복으로세지않습니다.
- reported_followup_summaries.json: 대화에붙여준입력기여도/PCA입력비교/pair모델요약을전사했습니다. 원본서버CSV를수신한것이아닙니다.
- provenance.json: 보존파일SHA-256과범위. 파일무결성확인이원시예측재평가나재학습검증을뜻하지않습니다.

새학습/예측재생성/원시데이터재평가는하지않았고, 불리한seed나화합물을제외해주결과를교체하지않았습니다. 선택epoch수와로그의보고수치대조만검사했습니다. 과거시험관찰이있는탐색실험입니다.
