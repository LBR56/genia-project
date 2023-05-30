# genia-project

유튜브 인강 자막 크롤링 및 텍스트 분석

 천재교육은 인기 또는 비인기 강의를 찾아서 언어적 차이를 확인할 것입니다.  
이때, 원시데이터는 유튜브의 강의를 사용할 것이고, 각 모델에 맞는 전처리를 통해, 모델을 제작할 것입니다.  
최종적으로 모델이 나타내는 분류의 이유를 판단해보고, 인사이트를 도출할 것입니다.

이 레파지토리는 다음이 포함되어 있습니다.

```bash
.
└── docs # 임의의 메모 정보를 업데이트

1 directory
```

각 파일에 대한 간단한 설명과 실행법을 나열합니다.

## 목차

- [genia-project](#genia-project)
  - [목차](#목차)
  - [설치](#설치)
    - [.env 파일 수정](#env-파일-수정)
  - [사용법](#사용법)
  - [팀원 소개](#팀원-소개)
  - [Ref](#ref)

## 설치

가장 먼저 설치할 컴퓨터에 git을 복제합니다.

```bash
git clone https://github.com/LBR56/genia-project
```

이 프로젝트는 python 3.10.6 버전을 사용하였습니다.  
또 필요한 라이브러리는 requirements.txt에 기재되어 있습니다.

그러므로 필요한 라이브러리를 설치합니다.

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

### .env 파일 수정

***

기본적으로 config 파일은 다음의 모습을 가지고 있습니다.

```dosini
HOST="~"
USER="~"
DATABASE="~"
PASSWORD="~"
```

다음 각 요소를 변경해주지 않는다면, 원치 않는 결과가 나타날 위험이 있습니다.

## 사용법

1. ```python main.py```를 실행합니다.

```bash
python main.py
```

## 팀원 소개

- 이병률
- 이재영
- 정우찬
- 최난경

## Ref

- 천재교육 제니아 아카데미
