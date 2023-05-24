# 🏆 Level 2 Project # 1 :: KLUE 문장 내 개체간 관계 추출

### 📜 Abstract
> 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 인공지능 만들기. 

<br>



## 🎖️Project Leader Board 
![public_7th](https://img.shields.io/static/v1?label=Public%20LB&message=7th&color=black&logo=naver&logoColor=white") 
![private_8th](https://img.shields.io/static/v1?label=Private%20LB&message=8th&color=black&logo=naver&logoColor=white")
- Public Leader Board
<img width="1089" alt="public_leader_board" src="https://github.com/boostcampaitech5/level2_klue-nlp-08/assets/81630351/0347231e-97ed-4329-bdf3-f0b23d51fbc2">

- Private Leader Board 
<img width="1089" alt="private_leader_board" src="https://github.com/boostcampaitech5/level2_klue-nlp-08/assets/81630351/25b8b164-7cd5-44dc-8c3b-9649a79d1bc7">

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

<br>

## 🗓️ Project Procedure

> *아래는 저희 프로젝트 진행과정을 담은 Gantt차트 입니다. 

<img width="959" alt="Screenshot 2023-05-24 at 3 31 48 PM" src="https://github.com/boostcampaitech5/level2_klue-nlp-08/assets/81630351/324e73cd-a16f-4a53-a76f-e039f5698360">

<br>

## 📁 Project Structure

### 📄 디렉토리 및 코드 구조 설명
> 학습 진행하기 전 증강 데이터 활용시 Augmentation을 학습 전에 진행<br>TAPT 적용시 TAPT 코드를 사전에 먼저 학습시켜 모델에 활용

- Augmentation : 데이터 증강 디렉토리
  - `augment_train.py` : 데이터 증강 모델 학습
  - `augment_dataloader.py` : 데이터 증강 모델 학습시 사용하는 dataloader
  - `augment.py` : 데이터 증강 코드
- dataset : 학습/테스트 데이터 디렉토리
  - `train/train.csv` : 학습 데이터 
  - `test/test_data.csv` : 테스트 데이터
- config : 모델 학습, 추론, 증강에 관련된 설정을 담고 있는 파일
  - `augment.yaml` : 증강 관련 설정 파일. 
  - `default.yaml` : 모델 학습, 추론 관련 설정 파일. 다양한 모델, 하이퍼 파라미터 세팅
  - `ensemble.yaml` : 앙상블 설정 파일 (Hard Voting, Soft Voting, F1-score Weighted Voting)
  - `tapt.yaml` : TAPT 관련 설정 파일
- model_ensemble : 앙상블 실행 파일
  - `ensemble.py` : 앙상블 실행 코드
  - `ensemble_model.py` : 앙상블 기법 정의(Hard Voting, Soft Voting, F1-score Weighted Voting)
  - `utils.py` : 앙상블에 필요한 argmax, softmax 함수 정의
- models : 
  - `RBERT.py`: R-BERT 모델 정의 코드
  - `TAEMIN_CUSTOM_RBERT.py`: R-BERT 단순화 모델 정의 코드
  - `TAEMIN_R_RoBERTa.py`: R-Roberta 모델 파일
  - `TAEMIN_RoBERTa_LSTM.py`: Roberta-LSTM 모델 정의 코드
  - `TAEMIN_TOKEN_ATTENTION_BERT.py`: BERT + CLS Token Attention 모델 정의 코드
  - `TAEMIN_TOKEN_ATTENTION_RoBERTa.py`: Roberta + CLS Token Attention 모델 정의 코드
  -  `model_base.py`: base 모델 정의 코드(FC Layer, RobertaClassificationHead, RobertaPooler)
  -  `utils.py`:
- modules : 모델에 쓰이는 dataset, loss 등 세부적인 모듈 정의 디렉토리
  - `datasets.py `: 모델 별 dataset 생성 코드
  - `losses.py` : Focal loss 코드
  - `optimizers.py` : AdamW, Adam, SGD, Adabelief 등 Optimizer 정의 코드
  - `preprocess.py` : 데이터 파싱 및 전처리 코드
  - `schedulers.py` : StepLR, CosinLR 정의 코드
  - `tokenize.py` : 토크나이징 및 Entity Representation 코드
  - `utils.py` : micro_f1, config_parser, confusion_matrix 코드
- pickle : 숫자 label - 스트링 label 변환 피클 파일
  - `dict_label_to_num.pkl` : 숫자 label을 스트링 label로 변환하는 피클 파일
  - `dict_num_to_label.pkl `: 스트링 label을 숫자 label로 변환하는 피클 파일
- prediction : 모델 추론 저장 디렉토리
- tapt : 
  - `dataset.py` : TAPT 데이터 로더 코드
  - `tapt.py` : TAPT 학습 코드
- .gitignore : gitignore
- `dataloader.py` : 모델 data loader 코드
- `inference.py` : 모델 추론 코드
- `model.py` : pytorch-lightning을 이용한 기본 모델 정의 코드
- `requirements.txt` : 환경 설정 관련 text파일
- `train.py` : 모델 학습 코드
- `wandb_tuning.py` : 여러개의 하이퍼 파라미터를 이용하여 wandb로 튜닝

```bash
📦level2_klue-nlp-08
 ┣ augmentation
 ┃ ┣ augment.py
 ┃ ┣ augment_dataloader.py
 ┃ ┣ augment_train.py
 ┃ ┗ utils.py
 ┣ config
 ┃ ┣ augment.yaml
 ┃ ┣ default.yaml
 ┃ ┣ ensemble.yaml
 ┃ ┗ tapt.yaml
 ┣ dataset
 ┃ ┣ test
 ┃ ┃ ┗ test_data.csv
 ┃ ┣ train
 ┃ ┃ ┗ train.csv
 ┣ model_ensemble
 ┃ ┣ ensemble.py
 ┃ ┣ ensemble_model.py
 ┃ ┗ utils.py
 ┣ models
 ┃ ┣ RBERT.py
 ┃ ┣ TAEMIN_CUSTOM_RBERT.py
 ┃ ┣ TAEMIN_R_RoBERTa.py
 ┃ ┣ TAEMIN_RoBERTa_LSTM.py
 ┃ ┣ TAEMIN_TOKEN_ATTENTION_BERT.py
 ┃ ┣ TAEMIN_TOKEN_ATTENTION_RoBERTa.py
 ┃ ┣ model_base.py
 ┃ ┗ utils.py
 ┣ modules
 ┃ ┣ datasets.py
 ┃ ┣ losses.py
 ┃ ┣ optimizers.py
 ┃ ┣ preprocess.py
 ┃ ┣ schedulers.py
 ┃ ┣ tokenize.py
 ┃ ┗ utils.py
 ┣ pickle
 ┃ ┣ dict_label_to_num.pkl
 ┃ ┗ dict_num_to_label.pkl
 ┣ prediction
 ┣ tapt
 ┃ ┣ dataset.py
 ┃ ┗ tapt.py
 ┣ wandb
 ┣ .gitignore
 ┣ README.md
 ┣ dataloader.py
 ┣ inference.py
 ┣ model.py
 ┣ requirements.txt
 ┣ train.py
 ┗ wandb_tuning.py
```


<br>

## ⚙️ Architecture

|분류|내용|
|:--:|--|
|모델|[`klue/bert-base`](https://huggingface.co/klue/bert-base), [`klue/roberta-large`](https://huggingface.co/klue/roberta-large), [`ainize/klue-bert-base-re`](https://huggingface.co/ainize/klue-bert-base-re) `HuggingFace Transformer Model`+`Pytorch Lightning`활용 + Attention Layer or FC Layer|
|전처리|• `Entity Representation` : Entity marker / Typed entity marker / SUB,OBJ marker / punct(한글) 등 다양한 entity representation을 적용하여 최적의 성능을 내는 entity representation 적용 |• Evaluation 단계의 피어슨 상관 계수를 일차적으로 비교<br>• 기존 SOTA 모델과 성능이 비슷한 모델을 제출하여 public 점수를 확인하여 이차 검증|
|데이터|• `raw data` : 기본 train 데이터 32470개 <br>• `증강데이터` : MLM kue/roberta-large 모델을 활용하여 증강데이터를 만들고 uniform 분포를 만들어 총 53873개|
|검증 전략|• 만들었던 모델의 Validation 데이터를 inference에 Micro F1-Score와 AUPRC Score 비교 <br>• 최종적으로 리더보드에 제출하여 모델 성능 검증|
|앙상블 방법|• Entity Represenatation과 모델기법, 증강데이터 중 가장 성능이 좋은 모델 3개를 선정하여 soft voting 앙상블을 진행
|모델 평가 및 개선 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|. MLM 모델을 사용하여 데이터 증강을 실시하여 label imbalance 문제를 해결한다. 또한, Entity Representation을 활용하여 데이터를 전처리하고 HuggingFace 모델에 Attention layer와 FC Layer등을 추가하는 등 다양한 기법을 활용하여 모델 성능을 개선한다.|

<br>

## 💻 Getting Started

### ⚠️  How To install Requirements
```bash
#필요 라이브러리 설치
pip install -r requirements.txt
```

### ⌨️ How To Train 

```bash
# 데이터 증강 [optional]
python3 augmentation/augment.py --config=config/augment.yaml
python3 augmentation/augment_train.py --config=config/augment.yaml
# TAPT 학습 모델 생성 [optional]
python3 tapt/tapt.py --config=config/tapt.yaml
# train.py 코드 실행 : 모델 학습 진행
python3 train.py --config=config/default.yaml
```
### ⌨️ How To Infer output.csv

```bash
# 모델 예측 진행
python3 inference.py --config=config/default.yaml
# 앙상블 진행 [config를 통해서 option 선택]
python3 model_ensemble/ensemble.py --config=config/ensemble.yaml
```
