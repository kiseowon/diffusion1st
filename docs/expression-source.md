# 발현 입력의 AIVIVE 출처와 RMA 전처리

## 새로 확인된 연결

사용자 서버의 vitro_train_test.csv와 vivo_train_test.csv가 [AIVIVE Zenodo 배포 자료](https://doi.org/10.5281/zenodo.14984579)의 동명 파일과 MD5가 정확히 일치한다고 보고됐습니다. 사용자가 전달한 [논문](https://doi.org/10.1093/toxsci/kfaf100)의 Methods는 oligo를 통한 RMA 정규화를 설명하고, 로컬 AIVIVE README는 해당 Zenodo 입력을 RMA-normalized 데이터라고 명시합니다.

| 파일 | Zenodo 크기(bytes) | Zenodo와 로컬에서 일치한 MD5 |
| --- | ---: | --- |
| vitro_train_test.csv | 1,876,727,611 | 139f24ef70d1af8730f3e08ae032d723 |
| vivo_train_test.csv | 3,740,796,430 | 217681c2335da3371182588ec1ff5a33 |

따라서 현재 출처 기록은 **AIVIVE가 배포한 RMA 정규화 발현 자료를 입력으로 사용**한 것으로 갱신합니다. 파일명이나 값 범위만으로 추정한 것이 아니라, 배포 메타데이터 checksum과 로컬 checksum의 일치에 근거합니다. [기계 판독 기록](../results/source_verification_20261104/aivive_zenodo_checksum.json)을 함께 보존했습니다.

검사는 사용자 서버에서 수행됐고 이 환경에서는 제공된 출력만 확인했습니다. 원본 수GB CSV를 내려받아 별도로 hash를 계산하거나 raw CEL에서 RMA를 다시 실행하지 않았습니다. 논문 원본의 RMA 세부 버전·옵션 전체를 재현했다는 뜻은 아닙니다.

## 우리가 추가한 처리

```text
Open TG-GATEs 원시 마이크로어레이
    ↓ AIVIVE의 RMA 등 상위 처리
Zenodo 배포 CSV ← 사용자 서버 로컬 CSV와 checksum 일치
    ↓ 고정 probe 목록·보정 메타데이터로 연결
화합물·용량·단위·시간·등급별 replicate 평균 → prepared
    ↓ 대조군 target을 포함하는 pair 구성
prepared_controls
    ↓ 실제 처리군−매칭 대조군 반응
현재 실험의 source/target 변화량
    ↓ 학습 자료에서만 표준화·PCA·Ridge·잔차 압축
MLP / cGAN / DDPM의 반응 예측
```

보존한 conditioned_ivive.py 준비 코드는 data root의 `{vitro,vivo}_train_test.csv`를 읽습니다. prepared_controls/preparation.json은 prepared를, prepared_delta/preparation.json은 prepared_controls를 source로 기록합니다. 현재 반복 실행기는 prepared_delta를 읽는 대신 prepared_controls로부터 같은 형태의 변화량을 코드 안에서 구성합니다. 이 확인 작업에서 조건 평균을 재생성한 것은 아니며 준비 코드와 기존 준비 재생성 검사 기록을 연결해 설명합니다.

**RMA와 학습용 StandardScaler는 다른 단계**입니다. RMA는 상위 마이크로어레이 전처리이고 StandardScaler는 학습 source의 probe별 척도 조정입니다. RMA가 확인됐다고 학습 표준화가 불필요해지거나, 반대로 RMA를 다시 해야 하는 것은 아닙니다. 추가 log 변환이나 재정규화를 이번 확인 때문에 넣지 않습니다.

## 해석의 경계

RMA 정규화 출처와 log2 fold change, 통계적으로 유의한 DEG 판정은 구분해야 합니다. RMA는 일반적으로 log2 발현을 산출하지만, 본 기록의 직접 증거는 배포 CSV 동일성과 RMA 출처입니다. 임계값0.5/1.0은 현재 보조 평가의 진폭 기준이며 유의성 검정이 아니고, 어떤 평균과 fold-change 정의를 쓰는지 설명하지 않은 채 단순히2배 변화 판정이라고 부르지 않습니다.

corrected metadata의 과거 생성 버전, reference 취득 이력, probe 목록 선정의 원논문 상세, 과거 SVD64의 출처는 이 두 CSV checksum 검사로 해결되는 사항이 아닙니다. 현재 모델에 쓰지 않는 과거 SVD64와 현재 입력 발현의 출처를 혼동하지 않습니다. 또한 AIVIVE가 배포할 때 사용한80/20 분할을 그대로 재현한 것이 아니라, 우리 실험은 별도의 화합물 분할/5-fold를 사용합니다.
