# 무작위 출력 probe 256개 비교

현재 완료된 24h·High toy를 reference로 사용한다. 화합물 수나 입력 probe를 줄이지 않고, **예측 출력만 3,453개 중 고정된 무작위 256개로 제한**한다. 기본 probe seed20261109로 복원 없이 뽑고 원래 probe 순서대로 정렬해 selected_probes.csv와 run.json에 보존한다. 이름·개수만 이용하며 반응 크기나 성능을 보고 목록을 고르지 않는다. 동일 목록을 모든 모델·구조·초기화에 사용한다. 이는 256개 유전자 선정이 아니라 256개 probe 선정이며 같은 유전자의 여러 probe가 포함될 수 있다.

## 통제 조건

기존 코호트134개와 train85/validation22/test27 분할을 그대로 재사용한다. 실제 코호트와 초기화 수는 reference에서 읽는다. 입력은 기존 source 전체 probe에서 적합한 PCA64와 구조 없음/Morgan/2D 표현을 유지한다. 저장된 inner/final 입력 전처리를 재사용하며 다시 fit하지 않는다. 출력의 probe별 학습 평균은 선택 열만 취하고 **공통 RMS는 기존 전체 출력의 값을 유지**한다. 따라서 정규화 설정도 맞춘 비교다. 이는 256개만 측정한 데이터 상황의 재현이 아니라, 모든 자료가 있는 상태에서 출력 크기를 줄이는 실험이다.

구조 없음/Morgan/2D × MLP/cGAN/DDPM × 기존 model seed42/43/44가 기본27조건, 최대54fit이다. 기존 전체 출력 모델은 재학습하거나 변경하지 않는다. 기존모델이 예측한 **같은256개 열만 추출해** 새256개 모델과 동일한 정답에서 재평가한다. 두 기준선도 같은 열에서 한 번씩 계산한다. 기존3,453개 전체 RMSE와 새256개 RMSE를 그대로 비교하지 않는다.

## Epoch와 모델 크기

각 arm/model/model seed의 epoch를 원래 전체 출력 모델의 validation 선택값으로 고정한다. 새 subset의 검증 RMSE는 해당 epoch에서 기록하되 이를 보고 새 epoch를 고르지 않는다. 최종 재학습 epoch도 동일하다. 이렇게 최대학습량과 선택 기준의 사후 변경을 제한한다. 원래 epoch는 전체probe 목적에서 선택됐으므로 새 subset에 최적인 epoch라고 주장하지 않는다.

입력은 같아도 출력 층·DDPM noisy-target 차원·cGAN 판별기의 출력 입력 차원과 총파라미터 수는 줄어든다. 손실도 선택된 probe만 포함한다. 같은 숫자 seed라도 네트워크 크기가 다르면 초기화·노이즈·배치 난수열이 완전히 같지는 않다. 따라서 출력 수와 모델 크기·학습 문제를 함께 바꾼 비교이고 probe 개수 하나의 인과 효과를 완전히 분리한 실험은 아니다. 모델 크기를 함께 저장한다.

## 평가와 해석

1차 비교는 paired_by_seed.csv의 subset_minus_full_rmse다. 음수면 같은 probe에서 subset 모델이 더 낮은 오차다. summary.csv는 두 출력학습 방식과 기준선의 RMSE/MAE/Pearson/Spearman/R²를, classification_summary.csv는 고정 임계값0.5/1.0의 F1 등을 보존한다. comparison_summary.csv는 모델·구조별 RMSE 대응 차이의 평균과 초기화 간 표준편차다. 표준편차를 신뢰구간이나 p-value로 해석하지 않는다.

한 분할·한 무작위 probe 집합의 탐색 파일럿이다. 화합물 수는 늘어나지 않고 원자료/시험에 대한 이전 관찰 이력도 남는다. 목록을 바꿔 가장 잘 나온 결과만 고르지 않는다. 이점이 보이면 사전에 정한 다른 무작위 목록으로 후속 반복을 별도 설계한다. 현재 결과만으로 최적 probe 수나 생물학적 유전자 선정의 우월성을 주장하지 않는다.

## 실행과 보존

`python scripts/random_output_subset_toy.py --reference-run /TOY/results --out /NEW_RESULTS --n-probes 256 --probe-seed 20261109 --execute`

--execute 없으면 계획만 출력한다. reference 완료 표지·입력/코드 hash·라이브러리 버전·probe 및 target 순서를 확인한다. 원본 파일을 바꾸지 않고 새 폴더에 기록한다. 같은 코드·reference·입력·옵션·출력으로 재개하면 완료 조건을 검사해 재사용한다. probe seed나 개수를 바꾸려면 새 출력 폴더가 필요하다.

합성 테스트는 고정 목록의 유일성/순서, 입력 및 출력스케일 보존, held-out 정답이 학습 Z에 들어가지 않음, 고정 reference epoch, 원본예측의 같은 열 재평가, 원본파일 불변·재개·목록변경 거부를 확인한다. 실제 서버 학습은 이 환경에서 실행하지 않았다.
