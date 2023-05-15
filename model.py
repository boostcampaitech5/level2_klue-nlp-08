import os

from adabelief_pytorch import AdaBelief

import pickle
import statistics
from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer,RobertaConfig)
from loss import FocalLoss
from model_list import TAEMIN_CUSTOM_RBERT, TAEMIN_TOKEN_ATTENTION_BERT, RBERT, TAEMIN_TOKEN_ATTENTION_RoBERTa,TAEMIN_R_RoBERTa,TAEMIN_RoBERTa_LSTM
from utils import klue_re_micro_f1, lr_scheduler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
focal_loss = FocalLoss(0.25,2.0)
class ERNet(pl.LightningModule):
    def __init__(self, config, wandb_config=None,state=None):
        super().__init__()

        if wandb_config == None:
            self.learning_rate = config["train"]["learning_rate"]
            self.weight_decay = config["train"]["weight_decay"]
        else:
            self.learning_rate = wandb_config.learning_rate
            self.weight_decay = wandb_config.weight_decay

        # self.model_config = AutoConfig.from_pretrained(config["model"]["model_name"])
        # self.model_config.num_labels = 30
        # self.model = AutoModelForSequenceClassification.from_pretrained(config["model"]["model_name"], config=self.model_config)

        roberta_config = RobertaConfig.from_pretrained("klue/roberta-large", num_laels=30)
        #
        self.model = TAEMIN_TOKEN_ATTENTION_RoBERTa.from_pretrained("klue/roberta-large", config=roberta_config,
                                                                    state=state).to(device)
        if state != 'train':
            self.model.resize_token_embeddings(32000 + 1)


        self.lr_scheduler_type = config["train"]["lr_scheduler"]

        self.train_step = 0

        self.validation_step_outputs = []
        self.output_pred = []
        self.output_prob = []

    def forward(self, x):
        x = self.model(**x)

        return x

    def configure_optimizers(self):
        optimizer = AdaBelief(self.parameters(), lr=self.learning_rate,weight_decouple=True,weight_decay=self.weight_decay)
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = lr_scheduler(lr_scheduler_type=self.lr_scheduler_type, optimizer=optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, _):
        y, x = batch.pop("labels"), batch
        y_hat = self(x).logits
        loss = F.cross_entropy(y_hat, y)
        #loss = focal_loss(y_hat,y)
        micro_f1 = klue_re_micro_f1(y_hat.argmax(dim=1).detach().cpu(), y.detach().cpu())
        self.log_dict({'train_micro_f1': micro_f1, "train_loss" : loss}, on_epoch=True, prog_bar=True, logger=True)
        if self.train_step % 100 == 0:
            print(f"learning_rate : {self.optimizers().optimizer.param_groups[0]['lr']}")
        self.train_step += 1
        return loss

    def on_train_epoch_end(self):
        self.train_step = 0

    def validation_step(self, batch, _):
        y, x = batch.pop("labels"), batch
        y_hat = self(x).logits
        loss = F.cross_entropy(y_hat, y)
        #loss = focal_loss(y_hat, y)
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
        output.to_csv('./prediction/submission.csv', index=False)

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
