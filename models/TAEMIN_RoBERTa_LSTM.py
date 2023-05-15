import torch
from torch import nn
from transformers import RobertaForSequenceClassification

from model_base import output_class


class TAEMIN_RoBERTa_LSTM(RobertaForSequenceClassification):
    def __init__(self, config=None,state=None):
        if config is None:
            config = RobertaForSequenceClassification.from_pretrained("klue/roberta-large",num_labels=30)
        super().__init__(config)
        if state == 'train':
            self.pretrained_model = RobertaForSequenceClassification.from_pretrained("klue/roberta-large", config=config)
            self.model_weights = self.pretrained_model.state_dict()
            self.load_state_dict(self.model_weights,strict=False)
            self.pretrained_model = None
        self.entity_embeddings = nn.Embedding(2, 1024)
        self.entity_embeddings.weight = nn.Parameter(torch.zeros(2, 1024))
        self.lstm = nn.LSTM(1024, 256, 1, batch_first=True)
        self.fc = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 30)
        self.dropout = nn.Dropout(0.1)
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
        output = torch.flip(outputs[0], dims=[1])
        output = self.dropout(output)
        _,(logits,_) = self.lstm(output)
        logits = logits.view(-1, 256)
        logits = self.fc(logits)
        logits = self.fc2(logits)
        return output_class(logits)
