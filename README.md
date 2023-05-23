# 🏆 Level 2 Project # 1 :: KLUE 문장 내 개체간 관계 추출

### 📜 Abstract
> 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 인공지능 만들기. 

<br>



## 🎖️Project Leader Board 
![public_7th](https://img.shields.io/static/v1?label=Public%20LB&message=7th&color=black&logo=naver&logoColor=white") ![private_8th](https://img.shields.io/static/v1?label=Private%20LB&message=8th&color=black&logo=naver&logoColor=white">)
- Public Leader Board
<img width="1089" alt="public_leader_board" src="https://github.com/boostcampaitech5/level2_klue-nlp-08/assets/96534680/eab58040-b8d6-4b15-a56a-bae5216a64ba">

- Private Leader Board 
<img width="1089" alt="private_leader_board" src="https://github.com/boostcampaitech5/level2_klue-nlp-08/assets/96534680/b91c30a2-27d8-4b3c-8737-e0ed0c4c5d62">

- [📈 NLP 08조 Project Wrap-Up report 살펴보기](https://github.com/boostcampaitech5/level2_klue-nlp-08/files/11539899/KLUE_Wrap-Up_Report_NLP-08_.pdf)

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

### 🧑🏻‍🔧 Members' Role
> - 모더레이터 외에도 Github 관리자를 두어 베이스라인 코드의 버전 관리를 원활하게 하고, 같은 분야라도 다른 작업을 진행할 수 있도록 분업을 하여 협업을 진행하였다.

| 이름 | 역할 |
| :---: | --- |
| **`강민재`** | **EDA**(`길이,레이블,토큰, entity 편향 확인`),**ErrorAnalysis,데이터증강**(`단순 복제, classinverse 관계 교체 증강`),**데이터전처리**(`subject,objectentity스페셜 토큰 처리`) |
| **`김태민`** | **모델 실험**(`Attention layer 추가 실험, Linear/LSTM layer 추가 실험, Loss, Optimizer 실험`), **데이터 전처리**(`Entity Representation – ENTITY, Typed Entity`) |
| **`김주원`** | **모델 튜닝, 프로젝트 매니저**(`노션관리, 회의 진행`), **EDA, 모델 앙상블**(`Hard Voting, Soft Voting, Weighted Voting`), **Error Analysis**(`Error Analysis 라이브러리 개발`) |
| **`윤상원`** | **Github 베이스라인 코드 관리**(`코드 리팩토링, 버그 픽스, 코드 버전 컨트롤`), **모델 실험**(`TAPT 적용`), **데이터 전처리**(`Entity Representation – ENTITY, Typed Entity`), **EDA**(`UNK 관련 EDA`), **모델 튜닝** |
| **`신혁준`** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | **EDA**(`데이터 heuristic 체크, Label 별 관계 편향 조사`), **데이터 증강** (`동일 entity start_idx, end_idx 교체, Easy Data Augmentation – SR 기반 증강, 클래스 Down Sampling`) |

<br>

## 🖥️ Project Introduction 


|**프로젝트 주제**| **문장 내 개체간 관계 추출**(KLUE RE): 문장의 단어(Entity)에 대한 속성과 관계를 예측하는NLP Task|
| :---: | --- |
|**프로젝트 구현내용**| 1. Hugging Face의 Pretrained 모델과KLUE RE 데이터셋을 활용해 주어진 subject, object entity간의 30개 중 하나의 relation 예측하는 AI 모델 구축 <br> 2. 리더보드 평가지표인 Micro F1-Score와AUPRC 높은 점수에 도달할 수 있도록 데이터 전처리(Entity Representation), 데이터 증강, 모델링 및 하이퍼 파라미터 튜닝을 진행 |
|**개발 환경**|**• `GPU`**: Tesla V100 서버 4개 (RAM32G) /Tesla V4 (RAM52G) /GeForce RTX 4090 로컬 (RAM 24GB) <br> **• `개발 Tool`**: PyCharm, Jupyter notebook, VS Code [서버 SSH연결], Colab Pro +, wandb|
|**협업 환경**|**• `Github Repository` :** Baseline 코드 공유 및 버전 관리 <br>**• `Notion` :** KLUE  프로젝트 페이지를 통한 역할분담, 대회 협업 관련Ground Rule 설정, 아이디어 브레인 스토밍, 대회관련 회의 내용 기록 <br>**• `SLACK, Zoom` :** 실시간 대면/비대면 회의|

## 📁 Project Structure

### 🗂️ 디렉토리 구조 설명 
- 학습 데이터 경로: `./data`
- 공개 Pretrained 모델 기반으로 추가 Fine Tuning 학습을 한 파라미터 경로
    - `./save_folder/kykim/checkpoint-7960`
    - `./save_folder/snunlp/checkpoint-31824`
    - `./save_folder/xlm_roberta_large/checkpoint-7960`
- 학습 메인 코드: `./train.py`
- 학습 데이터셋 경로: `./data/aug_train.csv`
- 테스트 메인 코드: `./infer.py`
- 테스트 데이터셋 경로: `./data/test.csv`

### 📄 코드 구조 설명

> 학습 진행하기 전 데이터 증강을 먼저 실행하여 학습 시간 단축

- **데이터 증강** Get Augmentation Data : `augmentation.py`
- **Train** : `train.py`
- **Predict** : `infer.py`
- **Ensemble** : `python esnb.py`
- **최종 제출 파일** : `./esnb/esnb.csv`

```
📦level1_semantictextsimilarity-nlp-11
 ┣ .gitignore
 ┣ config_yaml
 ┃ ┣ kykim.yaml
 ┃ ┣ snunlp.yaml
 ┃ ┣ test.yaml
 ┃ ┗ xlm_roberta_large.yaml
 ┣ data
 ┃ ┣ train.csv
 ┃ ┣ aug_train.csv
 ┃ ┣ dev.csv
 ┃ ┗ test.csv
 ┣ wordnet
 ┃ ┗ wordnet.pickle
 ┣ save_folde
 ┃ ┣ kykim
 ┃ ┃ ┗ checkpoint-7960
 ┃ ┣ snunlp
 ┃ ┃ ┗ checkpoint-31824
 ┃ ┗ xlm_roberta_large
 ┃   ┗ checkpoint-7960
 ┣ esnb
 ┃ ┗ esnb.csv
 ┣ output
 ┃ ┣ xlm_roberta_large.csv
 ┃ ┣ kykim.csv
 ┃ ┗ snunlp.csv
 ┣ .gitignore
 ┣ Readme.md
 ┣ augmentation.py
 ┣ dataloader.py
 ┣ esnb.py
 ┣ infer.py
 ┣ train.py
 ┗ utils.py
 ```
<br>

## 📐 Project Ground Rule
>팀 협업을 위해 개선점 파악을 위해 지난 NLP 기초 프로젝트 관련한 회고를 진행하였다. 회고를 바탕으로 프로젝트의 팀 목표인 “함께 성장”과 “실험 기록하기”를 설정했다. 그리고 목표를 이루기 위한 Ground Rule을 설정하여 프로젝트가 원활하게 돌아갈 수 있도록 팀 규칙을 정했다. 또한, 날짜 단위로 간략한 목표를 설정하여 협업을 원활하게 진행할 수 있도록 계획을 하여 프로젝트를 진행했다. 

- **`a. 실험 & 노션 관련 Ground Rule`**: 본인 실험을 시작할 때, Project 대시보드에 본인의 작업을 할당한 뒤 시작한다. 작업은 ‘하나의 아이디어’ 단위로 생성하고 Task, 진행상태를 표시한다. 작업이 마무리되면 실험 결과가 성능의 향상에 상관없이 ‘실험 대시보드’에 기록하고 상태를 완료 표시로 바꿔 마무리한다. 작업 단위로 관리하되 그 실험을 어떤 가설에서 진행하게 되었는지, 성공했다면 성공했다고 생각하는 이유, 실패했다면 실패한 원인에 대해 간략하게 정리한다.
- **`b. Commit 관련 Ground Rule`**: 
   1. **전체 main branch Pull Request 관련 Rule :** main branch에 대한 pull request는 Baseline Code를 업데이트할 때마다 진행한다. commit message에는 점수, 데이터, 버전 내용이 들어가도록 작성하고 push 한다
  2. **개인 Branch Commit 관련 Rule :** git commit & push는 코드의 유의미한 변화가 있을 때마다 진행한다. Commit message에는 코드 수정 내용(추가/변경/삭제)이 들어가도록 작성하고 push 한다.
- **`c. Submission 관련 Ground Rule` :** 하루 submission 횟수는 1인 2회씩 할당한다. 추가로 submission을 하고 싶으면 SLACK 단체 톡방에서 다른 캠퍼에게 물어봐 여유 횟수를 파악한 뒤 추가 submission을 진행한다. Submission을 할 때 다른 팀원이 어떤 실험의 submission인지 파악할 수 있도록 사용한 모델, 데이터, 기법, 하이퍼파라미터 등이 들어갈 수 있도록 Description을 기재한다.
- **`d. 회의 관련 Ground Rule` :** 전원이 진행하지 않는 Small 회의는 다양한 방식(Zep, Google Meet, Zoom)으로 진행하고 회의 내용을 기록한다.

<br>

## 🗓️ Project Procedure

*아래는 저희 프로젝트 진행과정을 담은 Gantt차트 입니다. 

![road_map](https://github.com/boostcampaitech5/level2_klue-nlp-08/assets/96534680/023681aa-b2c5-43f0-86f6-9fcc2599a5ef)

<br>

## ⚙️ Architecture
|분류|내용|
|:--:|--|
|모델|[`kykim/electra-kor-base`](https://huggingface.co/kykim/electra-kor-base), [`snunlp/KR-ELECTRA-discriminator`](https://huggingface.co/snunlp/KR-ELECTRA-discriminator), [`xlm-roberta-large`](https://huggingface.co/xlm-roberta-large)+ `HuggingFace Transformer Trainer`|
|데이터|• `v1` : swap sentence, copied sentence 기법을 적용하여 레이블 불균형을 해소한 데이터셋<br>• `v2` : KorEDA의 Wordnet 활용하여 Synonym Replacement 기법으로 증강한 데이터셋|
|검증 전략|• Evaluation 단계의 피어슨 상관 계수를 일차적으로 비교<br>• 기존 SOTA 모델과 성능이 비슷한 모델을 제출하여 public 점수를 확인하여 이차 검증|
|앙상블 방법|• 상기 3개의 모델 결과를 모아서 평균을 내는 방법으로 앙상블 수행|
|모델 평가 및 개선 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|토크나이징 결과 분석을 통해 max_length를 수정하여 모델 학습 시간을 절반 가량 단축할 수 있었다. 다양한 증강 및 전처리 기법을 통해 label imbalance 문제를 해결하여 overfitting을 방지하고 성능을 크게 향상시켰다. 또한, HuggingFace Trainer와 wandb를 사용하여 여러 하이퍼파라미터를 한층 더 편리하고 효율적으로 관리할 수 있었다.|

<br>

## 💻 Getting Started

### ⚠️  How To install Requirements
```bash
#필요 라이브러리 설치
# version 0.5
pip install git+https://github.com/haven-jeon/PyKoSpacing.git
# version 1.1
pip install git+https://github.com/jungin500/py-hanspell
pip install -r requirements.txt
sudo apt install default-jdk
```

### ⌨️ How To Train 
```bash
# 데이터 증강
python3 augmentation.py
# train.py 코드 실행 : 모델 학습 진행
# model_name을 kykim/electra-kor-base, snunlp/KR-ELECTRA-discriminator, xlm-roberta-large로 변경하며 train으로 학습
python3 train.py # model_name = model_list[0]
python3 train.py # model_name = model_list[1]
python3 train.py # model_name = model_list[2]
```

### ⌨️ How To Infer output.csv
```bash
# infer.py 코드 실행 : 훈련된 모델 load + sample_submission을 이용한 train 진행
python3 infer.py # model_name = model_list[0]
python3 infer.py # model_name = model_list[1]
python3 infer.py # model_name = model_list[2]
python3 esnb.py
```

