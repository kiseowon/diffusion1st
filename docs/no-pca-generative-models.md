# PCA 없는 입력·출력의 cGAN/DDPM 추가 실험

기존 no_pca_mlp 파일럿의 MLP2조건은 보존하고, `meeting_time_structure_pilot.py --input-space full --output-space full --families ddpm cgan --arms basic morgan --cache-diffusion-context`로 생성모델4조건만 새 폴더에서 실행한다. 처음에는 같은split seed42/fold0/model seed42를 사용한다. 5fold 전체 검증이 아니다.

시간별source3453probe/조건/mask슬롯과목표in vivo조건/Morgan표현, 학습MinMax입력, 가역평균중심화+공통RMS정답은MLP와같다. source특징은화합물별공유하고batch때concat한다. 정답을PCA로압축하거나probe를제거하지않는다.

## 모델·예산

- DDPM: context256/head64, noisy response3453 + timestep embedding16, cosine20step, x0-MSE, 생성4회평균.
- cGAN: context256/head64, noise16, 출력3453, 별도D context256/head64. G손실10full-probeMAE+2BCE, D조건부BCE. D1/G1 update. cycle없음. 생성4회평균.
- 예측모델Adam lr1e-4, D는SGD lr1e-4/momentum0.9. batch128, clip1, max100epoch. 원래probe공간validation RMSE로선택하고112개화합물로재학습한다.
- 4조건, 최대8fit. GPU이전이나새자료수집은하지않는다. CPU4threads기본이며큰입력층과cGAN별도D로인해MLP보다비용이클수있다. model_dimensions.json에서실제크기를확인한다.

AIVIVE를참고한소형적응모델이며공식AIVIVE재현이아니다. 초기설정의최적성이나이번예산에서의충분한수렴을보장하지않는다.

## DDPM 평가 시 조건 특징 재사용

기존코드는생성sample과확산step마다같은context encoder를반복계산했다. 새선택옵션 `--cache-diffusion-context`는eval모드에서원래batch의context를한번계산하고, 작은context벡터만sample수만큼복제해각step의head에제공한다. 큰원시입력벡터의sample복제도피한다. timestep·noisy response·노이즈schedule·난수소비순서·학습은변경하지않는다.

현재context는결정적인Linear/활성화층이다. 모의검사에서기존생성과허용오차내일치함을확인했다. 행렬계산batch크기가달라부동소수점반올림에작은차이가생길수있으므로bitwise일치를주장하지않는다. cGAN은원래부터예측시context를계산한후sample을복제한다. 새옵션은기본off이며기존계산도보존한다.

## 저장·비교

기존MLP폴더에이어쓰지않는다. 새코드/sourcehash가바뀌므로옛실행재개는옛snapshot으로해야한다. 생성모델결과는같은fold/seed의MLP와조건별로비교한다. 이전5fold평균과직접비교하지않는다. 기준선은같은target의zero response와훈련시간×등급평균이다.

학습완료조건마다예측/model/검증이력을보존하고results.csv/분류지표를갱신한다. 소프트60분예산은진행중인job을종료하지않는다. 일부완료인경우완성된6조건비교로제시하지않는다. 중단후동일command.json과새실험snapshot으로재개할수있다.

cloud에서는공유raw입력의4조건합성학습/출력/판별기저장/재개,DDPM context계산횟수와기존샘플링일치를검사했다. 실제연구데이터의학습은사용자서버에서실행한다.
