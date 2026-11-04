# 같은 시험 조건의 잠재 MLP·cGAN·DDPM 비교

현재의 Ridge+잠재 잔차 방식을 유지하고, `--families`로 cGAN을 추가했습니다. 단독 MLP/GAN/DDPM의 절대 발현 예측으로 바꾼 실험이 아닙니다. 최종 반응은 `공통 목표 조건 Ridge + λ×잠재 공간에서 복원한 보정`입니다. 과거 서버 실행 셀과 결과는 보존합니다.

## 공통 조건

같은 화합물 분할·내부 검증·source 변화량 PCA·Morgan+presence·조건 특징·출력 잔차 PCA를 공유합니다. 전처리·PCA·Ridge는 각 학습 집합에만 적합합니다. 모든 보정 모델이 구조를 쓰고 pathway는 없습니다. 반복·생성 seed·지표 집계는 [반복 검증 문서](repeated-latent-validation.md)를 따릅니다.

목표 조건 Ridge, 시간×등급 훈련 평균, 반응0을 같은 target에서 평가합니다. [기준선 설명](baselines.md)을 참고하세요. 기준선을 평가에서 빼거나 유리한 비교만 골라 보고하지 않습니다.

## 계열별 학습

| 모델 | 학습 | 평균 예측 |
| --- | --- | --- |
| 잠재 MLP | latent/noisy 입력0·time0을 고정한 잠재 MSE | 결정적 출력 |
| 잠재 cGAN | 조건부 적대적 BCE, 재구성 MSE 없음 | 정규 noise16차원, 16 sample 평균 |
| 잠재 x₀-DDPM | noisy latent와 time으로 깨끗한 latent를 예측하는 MSE | cosine30단계, 16 sample 평균 |

cGAN generator는 공통 입력→64 context, context+noise16→128 hidden→latent입니다. discriminator는 별도의 공통 입력→64 context와 실제/생성 latent를 결합해128 hidden(LeakyReLU)→logit1을 계산합니다. 두 context는 가중치를 공유하지 않고 각각 학습합니다. batch마다 discriminator와 generator를 한 번씩 업데이트합니다. discriminator는 실제1/생성0, generator는 생성1을 목표로 BCEWithLogits를 사용합니다.

AdamW lr0.0003, weight_decay0.01, clip1, batch64, 최대100epoch를 공통으로 사용합니다. cGAN의 두 optimizer에도 각각 적용합니다. 같은 검증 epoch와 보정 강도 후보에서 실제 반응 RMSE로 선택합니다. cGAN에 재구성 MSE나 별도 추가 튜닝을 넣지 않았습니다. 기본 손실의 탐색 비교이며 각 모델을 충분히 최적화했다는 뜻은 아닙니다.

같은 데이터와 선택 예산을 사용하지만 parameter 수·연산량까지 같은 비교는 아닙니다. GAN은 두 network를 학습하고 DDPM은 반복 추론합니다. max fit은 family별 학습 workflow 수이며 GAN의 G/D를 각각 fit1회로 세지 않습니다. 같은 model seed라도 network와 난수 사용이 달라 minibatch 순서까지 완전히 같다는 뜻은 아닙니다.

## 계획·실행

```bash
python scripts/repeated_latent_validation.py \
  --prepared /home/seowon1211/data1/conditioned_expression/prepared_controls \
  --structure /home/seowon1211/data1/conditioned_expression/morgan_matched/morgan_binary.csv \
  --out /새로운/three_family_결과폴더 \
  --families mlp cgan ddpm
```

기본은 계획 모드입니다. 5-fold×3 model seed×3 family=45모델 조건, 최대90회 신경망 fit입니다. `--execute`를 붙여야 실제 학습합니다. 기존 기본값 mlp/ddpm은 유지했으므로 세 family는 명시적으로 선택해야 합니다. family 선택은 run 계약에 포함됩니다. cGAN을 기존 완료 run에 끼워 넣지 말고 새 폴더로 실행하세요.

기존 커밋381cd48에서 완료한 run의 재개에는 그 고정 코드를 계속 사용합니다. 새 코드로 덮어쓰면 source/설정 계약이 달라 재개를 거부합니다. MLP/DDPM의 network와 선택 방식은 유지했지만 재실행 환경과 수치는 별도 기록해야 합니다.

집계에 cGAN을 포함하고 DDPM−cGAN, cGAN−MLP의 대응 차이를 추가합니다. 추가 평가 코드도 run의 families를 읽어 회귀·방향·F1을 평가합니다. families 필드가 없는 예전 run은 mlp/ddpm으로 해석합니다. bootstrap은 DDPM을 선택한 경우 사용 가능한 비교 모델을 대상으로 하며, 기존 고정 예측에 조건부인 구간이라는 한계를 유지합니다.

합성 검사에서는 GAN의 두 network 업데이트, 고정 seed 생성 평균, 세 family 계획 수, 축소한 합성 전체 실행·완료 수·대응 비교·추가 평가를 확인했습니다. 전체12개 검사 통과는 실제 연구 학습 재현이나 성능 개선 증거가 아닙니다. 코드 작성 중 연구 데이터 학습은 실행하지 않았습니다.
