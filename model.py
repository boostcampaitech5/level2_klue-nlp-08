import os
import pickle
import statistics
from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)
from adabelief_pytorch import AdaBelief
from model_list import CustomBertForSequenceClassification

from utils import klue_re_micro_f1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ERNet(pl.LightningModule):
    def __init__(self, learning_rate : float, weight_decay : float, model_name : str = "klue/bert-base",state=None):
        super().__init__()

        self.model_config = AutoConfig.from_pretrained(model_name)
        self.model_config.num_labels = 30
        self.model = CustomBertForSequenceClassification.from_pretrained(model_name, config=self.model_config,state=state).to(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.validation_step_outputs = []
        self.output_pred = []
        self.output_prob = []

    def forward(self, x):

        x = self.model(**x)
        #x = self.model(input_ids=x['input_ids'], token_type_ids=x['token_type_ids'],attention_mask=x['attention_mask'],index_ids=x['index_ids'])
        return x

    def configure_optimizers(self):
        optimizer = AdaBelief(self.parameters(), lr=self.learning_rate,weight_decouple=True,weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=1)
        return [optimizer], [scheduler]

    def training_step(self, batch, _):
        y, x = batch.pop("labels"), batch
        y_hat = self(x).logits
        loss = F.cross_entropy(y_hat, y)
        micro_f1 = klue_re_micro_f1(y_hat.argmax(dim=1).detach().cpu(), y.detach().cpu())
        self.log_dict({'train_micro_f1': micro_f1, "train_loss" : loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, _):
        y, x = batch.pop("labels"), batch
        y_hat = self(x).logits
        loss = F.cross_entropy(y_hat, y)

        pred = y_hat.argmax(dim=1)
        correct = pred.eq(y.view_as(pred)).sum().item()
        micro_f1 = klue_re_micro_f1(pred.detach().cpu(), y.detach().cpu()).item()

        preds = {"val_micro_f1": micro_f1, "val_loss" : loss, "correct" : correct}
        self.validation_step_outputs.append(preds)
        return preds

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_f1 = statistics.mean([x['val_micro_f1'] for x in self.validation_step_outputs])
        self.log_dict({'val_micro_f1': avg_f1, 'val_loss': avg_loss})
        print(f"{{Epoch {self.current_epoch} val_micro_f1': {avg_f1} val_loss : {avg_loss}}}")
        self.validation_step_outputs.clear()

    def test_step(self, batch, _):
        x = batch
        y_hat = self(x).logits
        prob = F.softmax(y_hat, dim=-1).detach().cpu().numpy()
        pred = y_hat.argmax(dim=1, keepdim=True)
        self.output_pred.extend(pred.squeeze(1).tolist())
        self.output_prob.extend(prob.tolist())
        return pred

    def on_test_epoch_end(self):
        pred_answer = self.num_to_label(self.output_pred)
        test_id = list(range(len(self.output_pred)))
        output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':self.output_prob})
        os.makedirs("prediction", exist_ok=True)
        output.to_csv('./prediction/submission_re.csv', index=False)

    def num_to_label(self, label : List[int]) -> List[str]:
        """
        숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
        """
        origin_label = []
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickle", 'dict_num_to_label.pkl'), 'rb') as f:
            dict_num_to_label = pickle.load(f)
        for v in label:
            origin_label.append(dict_num_to_label[v])
        
        return origin_label

