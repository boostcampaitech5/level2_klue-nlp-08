import torch
from torch import nn
from transformers import BertConfig, BertForSequenceClassification

from model_base import output_class


class TAEMIN_CUSTOM_RBERT(BertForSequenceClassification):
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
        self.cls_token_Linear = nn.Linear(768,768)
        self.word_token_Linear = nn.Linear(768,768)
        self.concat_Linear = nn.Linear(768*2,768)
        self.Tanh_act = nn.Tanh()

        self.fully_cls_dropout = nn.Dropout(0.1)
        self.fully_word_dropout = nn.Dropout(0.1)

        self.cls_word_dropout = nn.Dropout(0.1)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_ids=None, token_type_ids=None,attention_mask=None,index_ids=None, inputs_embeds=None,e1 = None,e2=None):
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
        word_token = outputs[0]

        word_token_idx = torch.zeros((word_token.size()[0], word_token.size()[1], 768)).to(self.device)
        word_token_idx[index_ids == 1] = 1
        word_token_idx = word_token_idx * word_token
        word_token_idx = torch.mean(word_token_idx,dim=1)
        word_token_idx = self.Tanh_act(word_token_idx)
        word_token_idx = self.fully_word_dropout(word_token_idx)
        word_token_tensor = self.word_token_Linear(word_token_idx)

        cls_token = self.Tanh_act(cls_token)
        cls_token = self.fully_cls_dropout(cls_token)
        cls_token = self.cls_token_Linear(cls_token)

        concat_cls_word = torch.cat((cls_token,word_token_tensor),dim=1)
        concat_cls_word = self.cls_word_dropout(concat_cls_word)
        concat_cls_word = self.concat_Linear(concat_cls_word)


        cls_token = self.dropout(concat_cls_word)
        logits = self.classifier(cls_token)

        output = (logits.view(-1, self.num_labels))

        return output_class(output)
