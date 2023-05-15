from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers import BertConfig, BertForSequenceClassification, RobertaForSequenceClassification, RobertaConfig,AutoConfig,RobertaModel

import torch
import torch.nn.functional as F

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

    def forward(self, features,add_features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
#        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x+add_features.squeeze())
        return x
class output_class:
    def __init__(self,output):
        self.logits = output

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        #torch.nn.init.xavier_uniform_(self.linear.weight)
    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)
def entity_average(hidden_output, e_mask):
    """
    Average the entity hidden state vectors (H_i ~ H_j)
    :param hidden_output: [batch_size, j-i+1, dim]
    :param e_mask: [batch_size, max_seq_len]
            e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
    :return: [batch_size, dim]
    """
    e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
    length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

    # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
    sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
    avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
    return avg_vector

class RBERT(BertForSequenceClassification):
    def __init__(self, config=None, state=None):
        if config is None:
            config = BertConfig.from_pretrained("klue/bert-base", num_labels=30)
            config.num_labels = 30
        super().__init__(config)
        if state == 'train':
            self.pretrained_model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=30)
            self.model_weights = self.pretrained_model.state_dict()
            self.load_state_dict(self.model_weights, strict=False)
            self.pretrained_model = None
        self.entity_embeddings = nn.Embedding(2, 768)
        self.entity_embeddings.weight = nn.Parameter(torch.zeros(2, 768))
        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            30,
            0.1,
            use_activation=True,
        )

    def forward(self, input_ids=None, token_type_ids=None,attention_mask=None,index_ids=None, inputs_embeds=None,e1 = None,e2=None):
        if inputs_embeds is None:
            inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)

        add_embeds = self.entity_embeddings(index_ids)
        inputs_embeds = inputs_embeds + add_embeds

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        #e1
        e_mask_unsqueeze = e1.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e1 != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), sequence_output).squeeze(1)
        e1_h = sum_vector.float() / length_tensor.float()
        #e2
        e_mask_unsqueeze = e2.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e2 != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), sequence_output).squeeze(1)
        e2_h = sum_vector.float() / length_tensor.float()

        # Average
        #e1_h = entity_average(sequence_output, e1)
        #e2_h = entity_average(sequence_output, e2)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        output = (logits.view(-1, 30))

        return output_class(output)
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

        word_token_idx = torch.zeros((word_token.size()[0], word_token.size()[1], 768)).to(device)
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


class TAEMIN_TOKEN_ATTENTION_BERT(BertForSequenceClassification):
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
        self.word_token_Linear_key = nn.Linear(768, 768)
        self.word_token_Linear_value = nn.Linear(768, 768)
        self.cls_toke_Linear_query = nn.Linear(768,768)
        self.entity_embeddings = nn.Embedding(2, 768)
        self.entity_embeddings.weight = nn.Parameter(torch.zeros(2, 768))
        self.query_dropout = nn.Dropout(0.1)
        self.key_dropout = nn.Dropout(0.1)
        self.value_dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, index_ids=None, inputs_embeds=None,e1=None,e2=None):
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
        word_token = outputs[0] # 나머지 토큰(16,245,768)

         # cls_token

        word_token_idx = torch.zeros((word_token.size()[0], word_token.size()[1], 768)) # (16,245,768) #(16,245)
        word_token_idx[index_ids == 1] = 1
        word_token_tensor = word_token * word_token_idx.to(device)

        cls_token_query = self.cls_toke_Linear_query(cls_token)
        word_token_key = self.word_token_Linear_key(word_token_tensor)
        word_token_value = self.word_token_Linear_value(word_token_tensor)

        cls_token_query = self.query_dropout(cls_token_query)
        word_token_key = self.key_dropout(word_token_key)
        word_token_value = self.value_dropout(word_token_value)

        query = cls_token_query #문맥정보 포함
        key = word_token_key # 집중해야될 토큰들 나머지 0인
        attn_scores = torch.matmul(query.unsqueeze(1), key.transpose(-1, -2))  # (16, 1, 246)
        attn_dist = torch.nn.functional.softmax(attn_scores, dim=-1)  # (16, 1, 246)
        value = word_token_value
        weighted_avg = torch.matmul(attn_dist, value)  # (16, 1, 768)


        logits = self.classifier(weighted_avg.to(device)+ cls_token.unsqueeze(1))

        output = (logits.view(-1, self.num_labels))

        return output_class(output)

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
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, index_ids=None, inputs_embeds=None):

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

        logits = self.classifier(cls_token.unsqueeze(1),weighted_avg.to(device))

        outputs = (logits.view(-1, 30))

        return output_class(outputs)

class TAEMIN_R_RoBERTa(RobertaForSequenceClassification):
    def __init__(self, config=None, state=None):
        if config is None:
            config = RobertaForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=30)
        super().__init__(config)
        if state == 'train':
            self.pretrained_model = RobertaForSequenceClassification.from_pretrained("klue/roberta-large",
                                                                                         config=config)
            self.model_weights = self.pretrained_model.state_dict()
            self.load_state_dict(self.model_weights, strict=False)
            self.pretrained_model = None
        self.entity_embeddings = nn.Embedding(2, 1024)
        self.entity_embeddings.weight = nn.Parameter(torch.zeros(2, 1024))
        #self.pooler = RobertaPooler(config)
        self.cls_fc_layer = FCLayer(1024, 512, 0.1)
        self.entity_fc_layer = FCLayer(1024, 512, 0.1)
        self.label_classifier = FCLayer(
            512 * 3,
            30,
            0.2,
        use_activation=False,
        )
        #self.tanh = nn.Tanh()
        #self.re_label_classifier = nn.Linear(512,30)
        #self.re_dropout = nn.Dropout(0,1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, index_ids=None, inputs_embeds=None,
                e1=None, e2=None):

        if inputs_embeds is None:
            inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        add_embeds = self.entity_embeddings(index_ids)
        inputs_embeds = inputs_embeds + add_embeds

        outputs = self.roberta(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
        )

        # outputs[0] # cls token (16,1,768)
        sequence_output = outputs[0]
        pooled_output = outputs[0]
        pooled_output = pooled_output[:, 0, :]# [CLS]
        #pooled_output = self.pooler(pooled_output)
        # e1
        e_mask_unsqueeze = e1.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e1 != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), sequence_output).squeeze(1)
        e1_h = sum_vector.float() / length_tensor.float()
        # e2
        e_mask_unsqueeze = e2.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e2 != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), sequence_output).squeeze(1)
        e2_h = sum_vector.float() / length_tensor.float()

        # Average
        # e1_h = entity_average(sequence_output, e1)
        # e2_h = entity_average(sequence_output, e2)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        return output_class(logits)

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


