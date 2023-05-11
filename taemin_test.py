from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers import BertConfig, BertForSequenceClassification, RobertaForSequenceClassification, RobertaConfig, \
    AutoConfig, RobertaModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, RobertaTokenizer
from model_list import TAEMIN_TOKEN_ATTENTION_BERT
import torch
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, AutoConfig,RobertaModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 30)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
class output_class:
    def __init__(self,output):
        self.logits = output
class TAEMIN_TOKEN_ATTENTION_RoBERTa(RobertaForSequenceClassification):
    def __init__(self, config=None,state=None):
        if config is None:
            config = RobertaForSequenceClassification.from_pretrained("klue/roberta-large",num_labels=30)
        super().__init__(config)
        if state == 'train':
            self.pretrained_model = RobertaForSequenceClassification.from_pretrained("klue/roberta-large", config=config)
            self.model_weights = self.pretrained_model.state_dict()
            self.load_state_dict(self.model_weights,strict=False)
            self.pretrained_model = None
        self.word_token_Linear_key = nn.Linear(1024, 1024)
        self.word_token_Linear_value = nn.Linear(1024, 1024)
        self.cls_toke_Linear_query = nn.Linear(1024, 1024)
        self.entity_embeddings = nn.Embedding(2, 1024)
        self.entity_embeddings.weight = nn.Parameter(torch.zeros(2, 1024))
        self.query_dropout = nn.Dropout(0.1)
        self.key_dropout = nn.Dropout(0.1)
        self.value_dropout = nn.Dropout(0.1)
        self.classifier = RobertaClassificationHead(config)
        #self.roberta_add_pooling_layer = RobertaPooler(config)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, index_ids=None, inputs_embeds=None,e1=None,e2=None):

        if inputs_embeds is None:
            inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        add_embeds = self.entity_embeddings(index_ids)
        inputs_embeds = inputs_embeds + add_embeds

        outputs = self.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids = token_type_ids
        )

        cls_token =  outputs[0][:, 0, :]#outputs[0] # cls token (16,1,768)
        #cls_token = self.roberta_add_pooling_layer(cls_token).unsqueeze(1)
        word_token = outputs[0]  # 나머지 토큰(16,245,768)

        # cls_token

        word_token_idx = torch.zeros((word_token.size()[0], word_token.size()[1], 1024))  # (16,245,768) #(16,245)
        word_token_idx[index_ids == 1] = 1
        word_token_tensor = word_token * word_token_idx.to(device)

        cls_token_query = self.cls_toke_Linear_query(cls_token)
        word_token_key = self.word_token_Linear_key(word_token_tensor)
        word_token_value = self.word_token_Linear_value(word_token_tensor)

        cls_token_query = self.query_dropout(cls_token_query)
        word_token_key = self.key_dropout(word_token_key)
        word_token_value = self.value_dropout(word_token_value)

        query = cls_token_query  # 문맥정보 포함
        key = word_token_key  # 집중해야될 토큰들 나머지 0인
        attn_scores = torch.matmul(query.unsqueeze(1), key.transpose(-1, -2))  # (16, 1, 246)
        attn_dist = torch.nn.functional.softmax(attn_scores, dim=-1)  # (16, 1, 246)
        value = word_token_value
        weighted_avg = torch.matmul(attn_dist, value)  # (16, 1, 768)

        logits = self.classifier(weighted_avg.to(device) + cls_token.unsqueeze(1))

        outputs = (logits.view(-1, 30))

        return output_class(outputs)



#config = AutoConfig.from_pretrained("klue/bert-base",num_labels=30)
#model = TAEMIN_TOKEN_ATTENTION_BERT.from_pretrained("klue/bert-base",config=config,state='train').to(device)

input_ids = torch.tensor([[31, 51, 99, 25, 10, 0, 0], [50, 12, 16, 6, 0, 0, 0]]).to(device)
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0]]).to(device)
token_type_ids = torch.tensor([[0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0]]).to(device)
index_ids = torch.tensor([[0, 0, 1, 0, 0, 0, 0,1,1,1,1,1,1,1,1,1]]).to(device)
e1 = torch.tensor([[1], [0]]).to(device)
e2 = torch.tensor([[1], [1]]).to(device)

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
text = "한국어 BERT 모델을 활용한 자연어 처리 예시입니다."
#model = RobertaForSequenceClassification.from_pretrained("klue/roberta-large").to(device)
config = RobertaConfig.from_pretrained("klue/roberta-large",num_laels=30)
model = TAEMIN_TOKEN_ATTENTION_RoBERTa.from_pretrained("klue/roberta-large",config=config,state='train').to(device)
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
token_type_ids = inputs["token_type_ids"].to(device)
print(model)
output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, index_ids=index_ids, e1=e1, e2=e2)
print(output.logits)
