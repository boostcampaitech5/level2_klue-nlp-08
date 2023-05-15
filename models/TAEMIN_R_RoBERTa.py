import torch
from torch import nn
from transformers import RobertaForSequenceClassification

from model_base import FCLayer, output_class


class TAEMIN_R_RoBERTa(RobertaForSequenceClassification):
    def __init__(self, config=None, state=None):
        if config is None:
            config = RobertaForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=30)
        super().__init__(config)
        if state == 'train':
            self.pretrained_model = RobertaForSequenceClassification.from_pretrained("klue/roberta-large",config=config)
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
