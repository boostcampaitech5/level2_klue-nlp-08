import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from models.RBERT import RBERT
from models.TAEMIN_CUSTOM_RBERT import TAEMIN_CUSTOM_RBERT
from models.TAEMIN_R_RoBERTa import TAEMIN_R_RoBERTa
from models.TAEMIN_RoBERTa_LSTM import TAEMIN_RoBERTa_LSTM
from models.TAEMIN_TOKEN_ATTENTION_BERT import TAEMIN_TOKEN_ATTENTION_BERT
from models.TAEMIN_TOKEN_ATTENTION_RoBERTa import \
    TAEMIN_TOKEN_ATTENTION_RoBERTa

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_model(model_name : str, state=None):
    print(f"이번 실험에서 사용되는 모델은 {model_name}입니다.")
    
    if model_name == "TAEMIN_TOKEN_ATTENTION_RoBERTa":
        model = TAEMIN_TOKEN_ATTENTION_RoBERTa.from_pretrained("klue/roberta-large", state=state)

    elif model_name == "TAEMIN_TOKEN_ATTENTION_BERT":
        model = TAEMIN_TOKEN_ATTENTION_BERT.from_pretrained("klue/bert-base", state=state)

    elif model_name == "TAEMIN_RoBERTa_LSTM":
        model = TAEMIN_RoBERTa_LSTM.from_pretrained("klue/roberta-large", state=state)

    elif model_name == "TAEMIN_R_RoBERTa":
        model = TAEMIN_R_RoBERTa.from_pretrained("klue/roberta-large",state=state)

    elif model_name == "TAEMIN_CUSTOM_RBERT":
        model = TAEMIN_CUSTOM_RBERT.from_pretrained("klue/bert-base", state=state)

    elif model_name == "RBERT":
        model = RBERT.from_pretrained("klue/bert-base", state=state)

    else:
        model_config = AutoConfig.from_pretrained(model_name, num_labels=30)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

    return model
