import torch
from torch import nn
from transformers import RobertaForSequenceClassification

from models.model_base import RobertaClassificationHead, output_class

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TAEMIN_TOKEN_ATTENTION_RoBERTa_NER(RobertaForSequenceClassification):
    def __init__(self, config=None,state=None):
        if config is None:
            config = RobertaForSequenceClassification.from_pretrained("SangwonYoon/klue-roberta-large-tapt",num_labels=30)
        super().__init__(config)
        if state == 'train':
            self.pretrained_model = RobertaForSequenceClassification.from_pretrained("SangwonYoon/klue-roberta-large-tapt", config=config)
            self.model_weights = self.pretrained_model.state_dict()
            self.load_state_dict(self.model_weights,strict=False)
            self.pretrained_model = None
        self.word_token_Linear_key = nn.Linear(1024, 1024)
        self.word_token_Linear_value = nn.Linear(1024, 1024)
        self.cls_toke_Linear_query = nn.Linear(1024, 1024)
        self.entity_embeddings = nn.Embedding(2, 1024)
        self.entity_embeddings.weight = nn.Parameter(torch.zeros(2, 1024))
        self.query_dropout = nn.Dropout(0.2)
        self.key_dropout = nn.Dropout(0.2)
        self.value_dropout = nn.Dropout(0.2)
        self.classifier = RobertaClassificationHead(config)
        self.ner_classifier = nn.Linear(1024,13)
        self.dropout_ner = nn.Dropout(0.2)
        #self.roberta_add_pooling_layer = RobertaPooler(config)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, index_ids=None, inputs_embeds=None,ner_list=None):

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
        word_token = outputs[0]  # 나머지 토큰(16,245,768)(전체 토큰임)

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

        logits_ner = self.dropout_ner(word_token + value)

        logits_ner = self.ner_classifier(logits_ner)
        logits = self.classifier(cls_token.unsqueeze(1),weighted_avg.to(device))

        outputs = (logits.view(-1, 30))
        outpus_ner = (logits_ner.view(-1, 13))
        return output_class(outputs,outpus_ner)
