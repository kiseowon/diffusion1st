# DDPM 은닉층·학습률·정규화 강도의 조합 탐색

`scripts/tune_ddpm_joint.py`는 외부 테스트를 평가하지 않는 내부 검증용 코드입니다. 한 파라미터의 최선값을 다른 조건의 최선값과 단순히 합치는 대신, 은닉층·학습률·weight decay를 조합으로 평가합니다. 기존 연구 결과와 원본 셀은 보존합니다.

## 사전 고정 후보와 두 단계

| 항목 | 후보 |
| --- | --- |
| context | 64 고정 |
| hidden | 64,128 |
| learning rate | 0.0001,0.0003,0.001 |
| weight decay | 0,0.01 |

총12조합입니다. 원래 reference는 context64·hidden128·lr0.0003·wd0.01이며, 직전 width 실험에서 좋았던 hidden64·lr0.0003·wd0.01도 후보에 포함합니다. 초기화 seed42에서12개 모두 학습하고 내부 검증 RMSE로 상위2개를 고릅니다. 이 후보들과 reference의 합집합에 seed43·44를 추가합니다. reference가 상위2개 안에 있으면 총16회, 없으면18회 내부 fit입니다. 기존 서버 run의 trial을 자동 재사용하지 않고 새 코드 계약으로 별도 실행합니다.

입력PCA 최대64·출력잔차PCA 최대32·cosine30step·16sample·batch64·clip1·max100epoch 및 epoch/λ 선택 규칙은 기존과 같습니다. 같은 전처리·PCA·기본 Ridge를 모든 후보가 공유하며 내부 학습에만 적합합니다. context64를 고정했으므로 모든 context/hidden/lr/wd의 전역 최적 탐색이 아닙니다.

## 분할과 선택

기본은 외부5-fold 중fold0,split seed42입니다. 화합물89개 내부 학습·23개 검증을 사용하고28개 외부 테스트는 점수 계산하지 않습니다. 로더는 전체 자료를 읽지만 외부 시험 정답은 학습·선택·결과 보고에 쓰지 않습니다.

각 trial에서 검증으로 epoch·보정 강도를 선택합니다. shortlist에 대해서만 세 model seed의 검증 RMSE 평균으로 최종 후보를 기록합니다. screen을 통과하지 않은 조합을 포함해 모든12개를3seed 평균으로 비교한 것은 아닙니다. selection은 같은 검증집합을 다시 사용하므로 점수가 낙관적일 수 있고, model seed3개를 독립 화합물 시험3개로 세지 않습니다. p-value나 최적성 확증을 만들지 않습니다.

## 실행

```bash
python scripts/tune_ddpm_joint.py \
  --prepared /home/seowon1211/data1/conditioned_expression/prepared_controls \
  --structure /home/seowon1211/data1/conditioned_expression/morgan_matched/morgan_binary.csv \
  --out /새로운/joint_validation_폴더
```

기본은 계획만 출력하며 입력 읽기·학습·출력 폴더 생성이 없습니다. 실제 학습에는 `--execute`를 추가합니다. 기본 CPU4threads입니다. `--screen-seed 42 --repeat-seeds 43 44 --top-k 2`가 기본이며 중복 seed는 거부합니다. outer fold·분할seed·후보·입력·source·환경을 계약에 기록합니다.

같은 계약과 출력 폴더로 재개하면 hash가 일치하는 완료 trial을 건너뜁니다. 실패한 trial은 새 attempt로 시작합니다. checkpoint나 파일을 외부 실행에 자동 적용하지 않습니다. 최종112개 refit과 외부 test 평가도 이 코드에서 하지 않습니다.

## 출력

- screening.csv: seed42의12조합 검증 점수와 선택 epoch/λ.
- shortlist.json: 상위2개+reference의 후보 목록.
- finalist_summary.csv: shortlist만의3seed 검증 평균·표준편차,reference 대비 대응 차이,parameter 수.
- trials.csv: 모든 실행 trial. screening에서만 평가한 후보는 seed1개뿐입니다.
- selected_candidate.json: shortlist 내에서 선택한 설정. 전역 최적이나 외부 시험 성능이 아닙니다.
- validation_history.csv,selection.json,best_validation_model.pt: trial별 기록. λ0 유지 시 checkpoint가 없을 수 있습니다.
- run.json,compound_split.csv,*audit.csv: 입력·코드hash·환경·분할·대조군·구조 연결 기록.
- completed.json: 실제 완료 trial 수와 outer_test_scored=false.

기존 `validate_ddpm_architecture.py`의 model factory는 후보에 learning_rate/weight_decay가 있으면 적용하도록 확장했습니다. 없는 경우 기존 기본값을 유지합니다. 과거 cd68878 width run의 재개에는 그 고정커밋을 사용해야 source 계약이 맞습니다.

## 해석의 경계와 검사

선택 후 일반화 성능을 평가하려면 각 outer fold 안에서 독립적으로 탐색·선택한 뒤 해당 test fold를 평가하는 nested 절차가 필요합니다. 한 fold 검증 결과로 전역 설정을 고르고, 같은 전체5-fold를 새로운 독립 시험처럼 보고하면 안 됩니다. DDPM만 추가 튜닝한 결과를 이전 기본 MLP/cGAN과 동등한 최적화 예산의 비교라고 표현하지 않습니다.

이번 추가 검사에서는12개 고유조합,계획모드 무실행,shortlist에 reference 유지 및 중복 제거,optimizer에 실제 후보값 적용,축소 합성자료의 두 단계 완료·동일seed범위·재개·설정 불일치 거부를 확인했습니다. 이는 연구 데이터에서의 최적성·재현 성능 검증이 아닙니다. 실제 연구 학습은 코드 작성 환경에서 실행하지 않았습니다.
