import torch
from torch import nn
from transformers import BertConfig, BertForSequenceClassification

from model_base import FCLayer, output_class


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
