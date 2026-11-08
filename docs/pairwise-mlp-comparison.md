# 한 source 조건씩 예측하는 MLP와 전체 source 연결 MLP 비교

목적은 in vitro 입력 방식 비교다. 기존 완료된 네 입력 기여도 실험을 reference로 사용한다. 목표조건만(A), 목표조건+Morgan(C), 여러source를flatten한B/D 결과를 유지하고 새 pair_basic/pair_morgan을 추가한다. 기존실험을 덮어쓰지 않는다. 기본reference는 분할seed42의5-fold·모델seed42/43/44로 신규30조건·최대60fit이며 원래 reference의 실제fold/seed수를 따른다.

## 입력/정답/전처리

한 source의 처리군−대조군 반응을 학습MinMax→PCA64→score표준화한 특징 + 해당source 시간/용량/단위/등급 + 목표in vivo 시간/용량/단위/등급 + 선택Morgan을 넣는다. 예측은 전체3453 probe의 목표반응이며, 최종학습평균/공통RMS 변환 후 역변환한다. source를단위별slot에따로두지않고 모든조건에동일MLP를적용한다. 입력에 화합물ID를주지않는다. pair는 prepared_controls의 기존같은화합물 조합 중 처리군target만 사용한다. 새pair선별/시간용량동일매칭/단위환산/생물학적지식/원자료다운로드는 하지 않는다. 개별replicate를복구하지않으며 기존조건평균을사용한다.

reference의present모드·같은fold/seed/구조에서 저장된 inner/final 전처리를 직접 재사용한다. scaler/PCA/target평균을 새로 fit하지 않는다. reference의파일hash·입력hash·주요라이브러리버전·공통코드hash를검사한다. validation/test정답은학습Z에넣지않고validation은target별평균예측RMSE로epoch선택에만쓴다. 새모델을최종학습112개화합물에서초기화해선택epoch만큼재학습한다. 조건별원래모델과선택epoch가같아야하는것은아니며선택규칙/최대학습budget이같다.

## 학습량과 가중치

목표t의source pair수가k_t이면각pair가중치는1/k_t다. 전체학습pair에서가중치평균이1이되도록정규화한다. minibatch별로가중치를재정규화하지않는다. 기준목적함수는 target별평균한 source별probe MAE의평균이다.

pair모든행을매epoch훑으면기존MLP보다업데이트수가많아지므로, 기본epoch는학습target수만큼전체pair에서균등복원추출한다. 역pair수가중치를함께적용하면위target-equal목적함수의불편gradient추정량이다. 같은최대epoch/batch128에서reference와최대optimizer업데이트수를맞춘다. 한epoch가모든pair를한번순회한다는뜻은아니고, 모든target이매epoch정확히한번관측된다는뜻도아니다. 실제pair수/target수/최대내부업데이트수/최종업데이트수를selection.json에기록한다. sample budget을늘리는옵션은이번실험에추가하지않는다.

평가에서는해당target의관측source pair를모두예측해원래probe공간에서평균한뒤target별1회평가한다. pair가늘어나도독립화합물수나평가target수가늘어나지않는다.

## 공정성의 범위

MLP context256/head64, Adam1e-4, MAE, batch128, clip1, 최대100epoch 등은reference와공유한다. 입력길이가달라첫층parameter수는달라진다. pair별loss와평가시평균을쓰므로기존flatten모델과학습목적함수도완전히동일하지않다. 입력표현/집계/목적함수의묶음비교이며pairing만의인과효과로주장하지않는다. AIVIVE의pair아이디어를참고하지만원논문의GAN/절대발현/replicate처리/최적화재현이아니다.

## 실행

`python scripts/pairwise_mlp_comparison.py --reference-run /INFORMATION_RUN/results --out /NEW_PAIR_RESULTS --execute`

--execute없으면계획만출력한다. reference완료표지가필요하다. 같은출력/코드/reference/입력/환경에서재개시완료조건을hash로확인하고재학습하지않는다. 입력및reference를변경하면거부한다. 실행코드와reference전처리를결과가끝날때까지보존한다.

summary.csv에는원래A/B/C/D·기준선과새pair2조건의전체OOF지표평균/std가있다. paired_rmse.csv는pair−flatten, pair−해당source없는조건, pair내Morgan−없음이며음수면앞방법오차가낮다. classification_summary.csv는임계값0.5/1.0의F1등을보존한다. 원본reference결과와새pair예측을합쳐평균/ensemble하지않는다. 같은5-fold를초기화3회반복한탐색평가이고독립분할3회/유의성검정/독립확증시험이아니다.

합성테스트는역pair수가중치에의한target별기여동일, source예측의target평균, heldout정답미사용, 전체OOF누락/중복검사, reference불변·재개·변경거부를확인한다. 실제학습은사용자서버에서진행하며합성테스트를실제재현완료로표현하지않는다.
