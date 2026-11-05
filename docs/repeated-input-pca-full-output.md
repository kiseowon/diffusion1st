# 입력 PCA 유지·전체 probe 출력의 고정 프로토콜 반복 검증

이번 반복 대상은 앞선 단일 holdout 전체 출력 파일럿이다. 출력 PCA를 사용한 이전 반복 결과와 구분하고 새 폴더에서 실행한다. 이 프로토콜을 데이터와 분리하여 새로 검증한 확증 실험은 아니며, 같은 테스트 화합물의 결과를 이미 관찰한 탐색적 후속 평가이다.

## 고정한 비교

- 구조 없음/basic, Morgan/morgan, bounded RDKit 2D Descriptor/descriptor
- MLP, cGAN, 조건부 x0-DDPM
- 화합물5fold, 분할seed42. 모델초기화seed42/43/44
- 9조건×5fold×3초기화=135조건, 내부학습후재학습까지최대270fit
- 각fold는내부89/검증23/테스트28개화합물. epoch선택후112개로전처리·모델재학습
- 같은화합물이학습/검증/테스트를넘지않으며모든모델은같은target을평가

입력은시간별source슬롯과관측mask를보존한다. 학습source의MinMax→PCA최대32→score표준화,출발/목표조건의dose log1p·시간표준화·단위/등급onehot,선택적구조를concat한다. 구조누락은기존정책대로0+presence0이다. Descriptor는signed-log·학습열선별·bounded MinMax를사용하며수치audit를보존한다. 출력은전체3453개probe반응이며학습probe평균과공통RMS만사용한다. 출력PCA·target clipping·probe제거는없다.

context256/head64, Adam lr1e-4, batch128, clip1, max100epoch. validation1/5/10/15/20/40/60/80/100의원래probe공간responseRMSE로선택한다. MLP는fullprobeMAE, cGAN은10MAE+2BCE와별도SGD판별기, DDPM은cosine20step의x0-MSE다. 생성4회평균. source/RMA를다시다운로드·정규화하거나SMILES를재검색하지않는다. AIVIVE설정일부를참고한자체적응프로토콜이며원본재현이나하이퍼파라미터최적성을주장하지않는다.

DDPM의 `--cache-diffusion-context`는eval시결정적인context를한번계산해sample/step에재사용한다. 학습·noise schedule·sample수는변경하지않으며이전계산과수치허용오차내동등성을모의검사했다. 계산batch크기가달라작은반올림차이가가능하다.

## 사전에 고정할 해석 기준

주지표는전체OOF target×probe의responseRMSE다. 보조로MAE/Pearson/Spearman/R2/임계값0.5·1의MacroF1및증가/감소F1을제시한다. 모델·표현별최고seed만고르지않고예정한초기화전체를보고한다. 기준선은zero response와훈련시간×등급평균이다.

짝지은차이는같은초기화에서Morgan−기본,Descriptor−기본,Descriptor−Morgan,각표현의DDPM−MLP/cGAN,각모델−두기준선을저장한다. 음수는앞방법의낮은RMSE다. 초기화3개와5fold를독립15개표본으로취급하지않는다. 표준편차는초기화변동이며화합물sampling CI·통계검정은아니다. F1은반응크기분류이며DEG유의성이나간독성라벨이아니다.

## 집계와 실행 범위

fold별RMSE를평균하지않고전체OOF예측을합쳐seed별평가한다. target이중복/누락되지않는지검사한다. 기준선은seed마다중복집계하지않고fold/seed간동일함을확인한다. summary.csv는seed별지표의평균/표준편차이며예측ensemble이아니다.

모델seed하나의5fold를먼저완료한다. 예산중단시5fold를완료한seed만전체OOF표에포함하고evaluation_status.json에완료범위를기록한다. 미완료seed는fold별표에만남긴다. progress.json에완료한fold/seed단위수와예상135조건을기록한다.

권장실행예산은전역120분/단위20분의소프트제한이다. 실행중인job을강제로중단하지않고최종집계도필요하므로120분내종료를보장하지않는다. 이전파일럿의시간으로정확한완료시간을예측할수없다. 중단시같은code snapshot/command.json/입력/환경/출력경로로재개하면완료학습은검사후건너뛴다. 기존실험폴더나코드snapshot에덮어쓰지않는다.

```bash
python scripts/repeat_time_structure_pilot.py \
 --prepared /home/seowon1211/data1/conditioned_expression/prepared_controls \
 --structure /home/seowon1211/data1/conditioned_expression/morgan_matched/morgan_binary.csv \
 --smiles /home/seowon1211/data1/compound_pubchem_smiles.csv \
 --out /home/seowon1211/data1/conditioned_expression/NEW_REPEATED_FULL_OUTPUT \
 --input-space pca --output-space full --cache-diffusion-context \
 --arms basic morgan descriptor --families mlp cgan ddpm \
 --folds 5 --split-seed 42 --model-seeds 42 43 44 \
 --epochs 100 --threads 4 --budget-minutes 120 --unit-budget-minutes 20
```

기본은입력을읽지않는계획모드이며실제학습에는 `--execute`를추가한다. cloud에서는전체출력9조건의축소합성반복/캐시옵션전달/OOF집계/기준선중복방지/짝지은Descriptor비교/재개를검사했다. 실제연구데이터의학습·성능확인은사용자서버에서수행한다.
