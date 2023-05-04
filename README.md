# 🏆 Level 2 Project # 1 :: KLUE 문장 내 개체간 관계 추출

### 📜 Abstract
> 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 인공지능 만들기. 

<br>

## 🧑🏻‍💻 Team Introduction & Members 

> Team name : 윤슬 [ NLP 08조 ] 

### 👨🏼‍💻 Members
강민재|김주원|김태민|신혁준|윤상원|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/39152134?v=4' height=130 width=130></img>|<img src='https://avatars.githubusercontent.com/u/81630351?v=4' height=130 width=130></img>|<img src='https://avatars.githubusercontent.com/u/96530685?v=4' height=130 width=130></img>|<img src='https://avatars.githubusercontent.com/u/96534680?v=4' height=130 width=130></img>|<img src='https://avatars.githubusercontent.com/u/38793142?v=4' height=130 width=130></img>|
<a href="https://github.com/mjk0618" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/Kim-Ju-won" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/taemin6697" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/jun048098" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/SangwonYoon" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a>
<a href="mailto:kminjae618@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:uomnf97@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:taemin6697@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:jun048098@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:iandr0805@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|

<br>

## 📐 Github Push Ground Rule
> 2대 Github Code Manager : 윤상원

1. git add 이전에 repository에 올라가면 안되는 파일들(데이터)이 `.gitignore`안에 들어있는지 확인하기
    -  만약 `.gitignore`에 없으면 파일을 추가해주세요. 
    - 기본적으로 data안에 들어가 있는 모든 파일들은 push 했을 때 remote 주소로 올라가지 않으니 train, test, dev 파일들은 생성시 data폴더를 만들어 해당 폴더 내부에 넣어주시는 걸 추천합니다.

2. `git commit` 이전에 본인의 branch가 맞는지 확인해주세요. (branch가 본인의 initial과 같은지 확인) 만약 아니라면 아래 명령어를 통해 본인의 브랜치로 반드시 변경해주세요.

```bash
# git checkout [본인브랜치 이름(이니셜)]
# 예시 
git switch -c KJW
```
```bash
# git checkout [본인브랜치 이름(이니셜)]
# 예시 
git switch KJW
```

3. **개인 Branch Commit 관련 Rule**  : `git commit & push`는 코드의 유의미한 변화가 있을 때 마다 진행합니다. 아래 양식을 보고 `코드 수정 내역(추가/변경/삭제)`이 들어가도록 commit 해주시면 됩니다. 처음에는 번거로울 수도 있지만, 협업을 위한 작업이니 변경시 commit을 꼭 부탁드립니다!

```bash
# git commit -am Upload/Update/Remove[코드 수정 사항][날짜]
# 예시 : 
git commit -am "Update Config Parser 23.04.11 18:50"
```

## 🗓️ 세부 일정 

- 프로젝트 전체 기간 (3주) : 5월 2일 (화) 10:00 ~ 5월 18일 (목) 19:00

  - 팀 병합 기간 : 5월 3일 (수) 16:00 까지

    - 팀명 컨벤션 : 도메인팀번호(2자리)조 / ex) CV_03조, NLP_02조, RecSys_08조

  - 리더보드 제출 오픈 : 5월 3일 (수) 10:00

  - 리더보드 제출 마감 : 5월 18일 (목) 19:00

  - 최종 리더보드 (Private) 공개 : 5월 18일 (목) 20:00

  - GPU 서버 할당 : 5월 2일 (화) 10:00

  - GPU 서버 회수 : 5월 19일 (금) 16:00

## 🖥️ 프로젝트 소개

문장 속에서 단어간에 관계성을 파악하는 것은 의미나 의도를 해석함에 있어서 많은 도움을 줍니다.

그림의 예시와 같이 요약된 정보를 사용해 QA 시스템 구축과 활용이 가능하며, 이외에도 요약된 언어 정보를 바탕으로 효율적인 시스템 및 서비스 구성이 가능합니다.
![image](https://user-images.githubusercontent.com/81630351/236123095-d45bf48d-00c2-42c5-94a5-443f0af08132.png)

> `관계 추출(Relation Extraction)`은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

이번 대회에서는 문장, 단어에 대한 정보를 통해 ,문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시킵니다. 이를 통해 우리의 인공지능 모델이 단어들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다. 우리의 모델이 정말 언어를 잘 이해하고 있는 지, 평가해 보도록 합니다.

```bash
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
subject_entity: 썬 마이크로시스템즈
object_entity: 오라클

relation: 단체:별칭 (org:alternate_names)
```
- **`input`**: sentence, subject_entity, object_entity의 정보를 입력으로 사용 합니다.

- **`output`**: relation 30개 중 하나를 예측한 pred_label, 그리고 30개 클래스 각각에 대해 예측한 확률 probs을 제출해야 합니다! 클래스별 확률의 순서는 주어진 dictionary의 순서에 맞게 일치시켜 주시기 바랍니다.

## 💯 평가 방법 : 

KLUE-RE evaluation metric을 그대로 재현했습니다.

1) no_relation class를 제외한 micro F1 score

2) 모든 class에 대한 area under the precision-recall curve (AUPRC)

2가지 metric으로 평가하며, micro F1 score가 우선시 됩니다.

### Micro F1 score

- micro-precision과 micro-recall의 조화 평균이며, 각 샘플에 동일한 importance를 부여해, 샘플이 많은 클래스에 더 많은 가중치를 부여합니다. 데이터 분포상 많은 부분을 차지하고 있는 no_relation class는 제외하고 F1 score가 계산 됩니다.

![image](https://user-images.githubusercontent.com/81630351/236133515-3bad534f-cf01-4234-8b59-df1e9a74b4d5.png)


### AUPRC

- x축은 Recall, y축은 Precision이며, 모든 class에 대한 평균적인 AUPRC로 계산해 score를 측정 합니다. imbalance한 데이터에 유용한 metric 입니다.
![image](https://user-images.githubusercontent.com/81630351/236133318-2c4fca5b-c14e-49bc-bd68-90f4b9af8b04.png)

- 위 그래프의 예시는 scikit-learn의 Precision-Recall 그래프의 예시 입니다. 그림의 예시와 같이 class 0, 1, 2의 area(면적 값)을 각각 구한 후, 평균을 계산한 결과를 AUPRC score로 사용합니다.
## 📁 프로젝트 구조

- Baseline 디렉토리 구조 
```bash
├── code
│   ├── __pycache__
│   ├── best_model
│   ├── logs
│   ├── prediction
│   └── results
└── dataset
    ├── test
    └── train
```
- Baseline 파일 포함 디렉토리 구조
```bash 
├── code
│   ├── __pycache__
│   │   └── load_data.cpython-38.pyc
│   ├── best_model
│   ├── dict_label_to_num.pkl
│   ├── dict_num_to_label.pkl
│   ├── inference.py
│   ├── load_data.py
│   ├── logs
│   ├── prediction
│   │   └── sample_submission.csv
│   ├── requirements.txt
│   ├── results
│   └── train.py
└── dataset
    ├── test
    │   └── test_data.csv
    └── train
        └── train.csv
```
### code 설명

- **train.py**
  - baseline code를 학습시키기 위한 파일입니다.
  - 저장된 model관련 파일은 results 폴더에 있습니다.

- **inference.py**
  - 학습된 model을 통해 prediction하며, 예측한 결과를 csv 파일로 저장해줍니다.
  - 저장된 파일은 prediction 폴더에 있습니다.
  - sample_submission.csv를 참고해 같은 형태로 제출용 csv를 만들어 주시기 바랍니다.

- **load_data.py** 
  - baseline code의 전처리와 데이터셋 구성을 위한 함수들이 있는 코드입니다.

- **logs**
  - 텐서보드 로그가 담기는 폴더 입니다.

- **prediction**
  - inference.py 를 통해 model이 예측한 정답 submission.csv 파일이 저장되는 폴더 입니다.

- **results**
  - train.py를 통해 설정된 step 마다 model이 저장되는 폴더 입니다.

- **best_model**
  - 학습중 evaluation이 best인 model이 저장 됩니다.

- **dict_label_to_num.pkl**
  - 문자로 되어 있는 label을 숫자로 변환 시킬 dictionary 정보가 저장되어 있습니다.

- **dict_num_to_label.pkl**
  - 숫자로 되어 있는 label을 원본 문자로 변환 시킬 dictionary 정보가 저장되어 있습니다.
  - custom code를 만드실 때도, 위 2개의 dictionary를 참고해 class 순서를 지켜주시기 바랍니다.

### dataset 설명

- **train**
  - train 폴더의 train.csv 를 사용해 학습 시켜 주세요.
  - evaluation data가 따로 제공되지 않습니다. 적절히 train data에서 나눠 사용하시기 바랍니다.

- **test**
  - test_data.csv를 사용해 submission.csv 파일을 생성해 주시기 바랍니다.
  - 만들어진 submission.csv 파일을 리더보드에 제출하세요

## 📏 대회 룰

- **[대회 참여 제한]** NLP 도메인을 수강하고 있는 캠퍼에 한하여 리더보드 제출이 가능합니다.

- **[팀 결성 기간]** 팀 결성은 대회 시작 2일차 화요일 오후 4시까지 필수로 진행합니다. 팀이 완전히 결성되기 전까지는 리더보드 제출이 불가합니다.

- **[일일 제출횟수]** 일일 제출횟수는 팀 단위 10회로 제한합니다. (일일횟수 초기화는 자정에 진행)

- **[외부 데이터셋 규정]** 본 대회에서는 **외부 데이터셋 사용**을 금지합니다. 학습에 사용될 수 있는 데이터는 제공되는 train.csv 한 가지 입니다

- **[평가 데이터 활용]** <U>**test_data.csv에 대한 Pseudo labeling을 금지합니다. test_data.csv을 이용한 TAPT(Task-Adaptive Pretraining)는 허용 합니다.**</U> 단 평가 데이터를 <U>**눈으로 직접 판별 후 라벨링 하는 행위**</U> 는 금지합니다. 제공된 학습 데이터을 사용한 데이터 augumentation 기법이나, 생성모델을 활용한 데이터 생성 등, 학습 데이터를 활용한 행위는 허용 됩니다.

- **[데이터셋 저작권]** 대회 데이터셋은 **'캠프 교육용 라이선스'** 아래 사용 가능합니다. **저작권 관련 세부 내용은 부스트코스 공지사항을 반드시 참고 해주세요.**

## AI Stages 대회 공통사항

- **[Private Sharing 금지]** 비공개적으로 다른 팀과 코드 혹은 데이터를 공유하는 것은 허용하지 않습니다.코드 공유는 반드시 대회 게시판을 통해 공개적으로 진행되어야 합니다.

- **[최종 결과 검증 절차]** 리더보드 상위권의 경우 추후 최종 코드 검수가 진행됩니다. 반드시 결과가 재현될 수 있도록 최종 코드를 정리해 주세요! 부정행위가 의심될 경우에는 결과 재현을 요구할 수 있으며, 재현이 어려울 경우 리더보드 순위표에서 제외될 수 있습니다.

- **[공유 문화]** 공개적으로 토론 게시판을 통해 모델링에 대한 아이디어 혹은 작성한 코드를 공유하실 것을 권장 드립니다. 공유 문화를 통해서 더욱 뛰어난 모델을 대회 참가자 분들과 같이 개발해 보시길 바랍니다.

- **[대회 참가 기본 매너]** 좋은 대회 문화 정착을 위해 아래 명시된 행위는 지양합니다.
  - 대회 종료를 앞두고 (3일 전) 높은 점수를 얻을 수 있는 전체 코드를 공유하는 행위
  - 타 참가자와 토론이 아닌 단순 솔루션을 캐내는 행위

<br>

## 💻 Getting Started

### ⚠️  How To install Requirements
```bash
#필요 라이브러리 설치
pip install -r requirements.txt
```

### ⌨️ How To Train 
```bash
python train.py
```
### ⌨️ How To Infer output.csv
```bash
python inference.py --model_dir = ‘모델 저장 경로’
#ex) python inference.py --model_dir=./results/checkpoint-500 
```

