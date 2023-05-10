from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers import BertConfig, BertForSequenceClassification

import torch
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class output_class:
    def __init__(self,output):
        self.logits = output
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        nn.CrossEntropyLoss()
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config=None,state=None):
        if config is None:
            config = BertConfig.from_pretrained("klue/bert-base",num_labels=30)
            config.num_labels=30
        super().__init__(config)
        if state == 'train':
            self.pretrained_model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=30)
            self.model_weights = self.pretrained_model.state_dict()
            self.load_state_dict(self.model_weights,strict=False)
            self.pretrained_model = None
        self.entity_embeddings = nn.Embedding(2, 768)
        self.entity_embeddings.weight = nn.Parameter(torch.zeros(2, 768))
    def forward(self, input_ids=None, token_type_ids=None,attention_mask=None,index_ids=None, inputs_embeds=None):
        # print('add',self.add_embeddings.weight)
        # print('token',self.bert.embeddings.token_type_embeddings.weight)
        if inputs_embeds is None:
            inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)

        add_embeds = self.entity_embeddings(index_ids)
        inputs_embeds = inputs_embeds + add_embeds

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_token = outputs[1] #cls token (16,1,768)

        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)

        output = (logits.view(-1, self.num_labels))

        return output_class(output)
