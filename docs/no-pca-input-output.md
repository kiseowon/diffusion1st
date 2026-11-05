# 입력·출력 PCA를 모두 제거한 MLP 파일럿

`meeting_time_structure_pilot.py --input-space full --output-space full`로 실행한다. 기존 PCA/전체출력 실험과 별도 폴더에 저장한다. 최초 범위는 구조 없음/basic과 Morgan의 MLP 두 조건, split seed42/fold0/model seed42이다. 이전 전체 출력 실험의 같은 fold/seed와 비교할 수 있지만, 아직 5fold 반복 결과가 아니다.

## 입력과 정답

각 관측 source 조건의 3,453개 probe 반응을 학습 고유 source로 적합한 MinMaxScaler로 변환한다. probe 제거·입력 PCA·입력 clipping은 하지 않는다. MinMax 이후 PCA score StandardScaler도 없어진다. 검증/시험의 값은 학습 범위를 벗어날 수 있다. 모든 시간·용량 슬롯과 관측 mask, source/target dose/time/unit/level, optional Morgan은 기존 정책을 유지한다. 단위 환산은 하지 않는다.

source 조건마다 `[3,453 probe 값, 조건 특징, mask]`를 슬롯에 배치해 펼친다. schema가36개일 때 probe 자리만124,308개이며 실제 총 입력 수는 조건/구조를 포함해 더 크다. 동일 화합물의 source 벡터는 target 조건마다 동일하지만 target 시간/용량 조건은 다르다.

정답은 전체3,453개 probe 반응이다. 학습 probe별 평균과 공통 RMS를 이용하는 가역 변환만 적용한다. 정답 probe 제거/출력 PCA/clipping은 없다. 신경망이 압축 없이 정답을 받는다는 뜻이며, 모델의 유한한 은닉층이 모든 패턴을 복원한다는 보장은 아니다.

## 메모리와 모델

`SharedSourceFeatures`는 화합물별 source 슬롯 벡터를 한 번 저장하고, target별로는 화합물 index와 목표 조건만 보존한다. batch를 요청할 때만 source/target/structure를 concat한다. target마다12만차원 source를 복사한 전체 입력 행렬을 만들지 않는다. 수치연산과 모델에 제공하는 벡터는 동일하며 관측 mask나 슬롯을 평균하지 않는다.

가중치는 여전히 큰 dense 입력층이다. 입력→256의 probe 연결만 약3,182만개이며 모델가중치/gradient/Adam state만 수백MB를 차지한다. cGAN에는별도판별기까지 필요하므로 우선 MLP2조건만 권장한다. `model_dimensions.json`은 각 내부/최종 단계의 실제 입력차원·출력차원·계열별명목parameter수·MLP학습상태의대략적메모리를저장하고시작전에출력한다. 메모리추정은활성값·데이터·batch·optimizer임시배열을포함하지않는다.

context256/head64, Adam lr1e-4, batch128, clip1, max100epoch를 유지한다. 출력은full probe MAE이며 원래반응공간validation RMSE로epoch를선택한다. 선택후112개외부학습화합물로전처리와모델을다시적합한다. source/target전처리는해당학습집합에만적합한다. source슬롯schema는기존처럼전체사용가능source메타데이터로만정한다.

## 해석과 실행

비교대상은 입력 PCA 최대32/출력full이었던 MLP 결과다. 구조없음과Morgan 각각동일fold/seed/target에서비교한다. **PCA제거와함께입력스케일링방식·입력층parameter수도달라진다.** 차이를PCA의정보손실효과하나로해석하거나, 한holdout결과를확증결론으로사용하지않는다. 화합물140개대비과적합위험이커진다.

기존파일은보존한다. 입력/소스/설정계약이다르면기존폴더재개를거부한다. `--budget-minutes 60`은새job시작을제한하는소프트예산으로진행중인학습을강제종료하지않는다. PCA보다오래걸릴수있으며RAM도필요하다. 예측품질개선과실행시간은검증되지않았다.

```bash
python scripts/meeting_time_structure_pilot.py \
 --prepared /home/seowon1211/data1/conditioned_expression/prepared_controls \
 --structure /home/seowon1211/data1/conditioned_expression/morgan_matched/morgan_binary.csv \
 --smiles /home/seowon1211/data1/compound_pubchem_smiles.csv \
 --out /home/seowon1211/data1/conditioned_expression/NEW_NO_PCA_MLP \
 --input-space full --output-space full \
 --arms basic morgan --families mlp --epochs 100
```

기본은plan-only이며`--execute`에서만학습한다. cloud에서는합성데이터로전체입력probe유지·화합물source공유·target전처리heldout미사용·두MLP학습·parameter계산일치를검사했다. 연구데이터학습은사용자서버에서실행한다. 추후반복실행기도`--input-space full --output-space full`을지원하지만옛PCA결과의자동병합은거부한다.
