# 구조 on/off 대조 결과

## 현재 결론

**동일한 고정 모델 조건에서 Morgan+presence를 제공했을 때 평균 반응 RMSE가 소폭 낮아지는 가능성을 확인했습니다.** 일반적 효과나 통계적 우월성을 입증한 결과는 아닙니다. 실제 KEGG/pathway의 효과를 검증한 실험도 아닙니다.

| 모델 | 구조 off 평균 RMSE | 구조 on 평균 RMSE | on−off | on이 낮은 model seed |
| --- | ---: | ---: | ---: | ---: |
| MLP | 0.291572 | 0.290797 | −0.000775 | 3/3 |
| DDPM | 0.291770 | 0.290991 | −0.000779 | 2/3 |
| cGAN | 0.294571 | 0.294099 | −0.000471 | 2/3 |

on/off 상대 개선은 각각 약0.27%,0.27%,0.16%입니다. 각 seed는5개fold 전체 target을 합산한 하나의OOF 결과입니다. 모든 개별 fold에서 개선됐다는 뜻이 아닙니다. 이 표는 seed별RMSE 평균이므로 기존 runner의MSE 평균 후sqrt 요약과 끝자리가 조금 다를 수 있습니다.

## 설계와 정상 완료 기록

[동일 조건 구조 대조 설계](../../docs/structure-ablation.md)를 따릅니다. 기존 on을 유지하고 Morgan bit와presence만0으로 만든off를 별도로 실행했습니다. 입력차원·parameter수·초기화seed·분할·발현/조건입력·공통Ridge·PCA·학습예산은 유지했습니다. 직전hyperparameter 튜닝 결과를 off에만 적용한 실험이 아닙니다.

서버 로그에는 paired on 계약 검사 통과,off45조건 완료,대응 비교 완료,추가 지표12unit 평가 완료가 기록돼 있습니다. 이 환경에서는 첨부 로그만 읽었으며 학습·추론을 하지 않았습니다. 실제서버 run.json·comparison.json·예측·모델·원본CSV를 받아 독립 재검사한 상태는 아닙니다.

## 추가 지표의 방향

이전에 보고된on 결과와 비교할 때,τ0.5의MacroF1은MLP37.058023→38.124653%,DDPM37.041325→37.858143%로 높았습니다. cGAN은36.072844→35.899277%로 소폭 낮았습니다. 정확도와F1은0.5/1.0의탐색적진폭분류이지유의한DEG 판정이아닙니다.

지표가 모두 개선된 것은 아닙니다. MLP의MAE는off0.173167,on0.173403으로off가 낮고,DDPM의MAE는off0.173199,on0.173203으로거의같습니다. DDPM/MLP의Spearman도off가소폭높습니다. 따라서RMSE 개선만으로생물학적반응의모든측면이좋아졌다고설명하지않습니다.

구조on의가장낮은평균RMSE도시간×등급평균기준선0.288403을넘지못했습니다. DDPM의cGAN대비이점은on/off모두보였지만,DDPM이MLP보다전반적으로좋았던것은아닙니다.

## 보존 자료와 출처

- [첨부 서버 로그 원본](server_console_output.txt): 업로드된바이트그대로보존.
- [구조 비교 요약 전사본](reported_structure_summary.csv).
- [seed별 구조 대응 비교 전사본](reported_structure_paired.csv).
- [off 회귀 지표 전사본](reported_off_regression_summary.csv).
- [off 방향·F1 전사본](reported_off_classification_summary.csv).
- [출처·확인범위](provenance.json).

`reported_`는원본서버CSV가아니라콘솔에표시된소수점6자리표를파싱해보존한파일이라는뜻입니다. 전사표내부의seed평균·개선seed수·offRMSE일치를반올림오차범위에서확인했습니다. 표시되지않은bootstrap CSV는평가하지않았고유의성결론을만들지않았습니다. 원본출력폴더는provenance.json에기록했습니다.
