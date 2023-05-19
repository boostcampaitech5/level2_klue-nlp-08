import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from modules.losses import get_loss
from modules.optimizers import get_optimizer
from modules.schedulers import get_scheduler
from modules.utils import klue_re_micro_f1, show_confusion_matrix, num_to_label

from models.utils import get_model

class ERNet(pl.LightningModule):
    def __init__(self, config, wandb_config=None, resize_token_embedding=None, state=None):
        super().__init__()

        if wandb_config == None:
            self.learning_rate = config["train"]["learning_rate"]
            self.weight_decay = config["train"]["weight_decay"]
        else:
            self.learning_rate = wandb_config.learning_rate
            self.weight_decay = wandb_config.weight_decay

        self.model = get_model(model_name=config["model"]["model_name"], state=state)
        if resize_token_embedding:
            self.model.resize_token_embeddings(resize_token_embedding)

        self.lr_scheduler_type = config["train"]["lr_scheduler"]
        self.optimizer_type = config["train"]["optimizer"]
        self.loss_type = config["train"]["loss"]

        self.train_step = 0

        self.confusion_matrix_path = config["path"]["confusion_matrix"]

        self.validation_step_outputs = []
        self.validation_preds = []
        self.validation_labels = []
        self.output_pred = []
        self.output_prob = []

    def forward(self, x):
        x = self.model(**x)

        return x

    def configure_optimizers(self):
        optimizer = get_optimizer(optimizer_type=self.optimizer_type)
        optimizer = optimizer(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_scheduler(scheduler_type=self.lr_scheduler_type)
        scheduler = scheduler(optimizer, step_size=1, gamma=0.3)
        return [optimizer], [scheduler]

    def training_step(self, batch, _):
        y, x = batch.pop("labels"), batch
        y_hat = self(x).logits
        y_hat_ner = self(x).ner_logits
        loss = get_loss(self.loss_type)
        loss_ner = get_loss(self.loss_type)
        loss = loss(label_smoothing=0.2)
        loss_ner = loss_ner(label_smoothing=0.2)
        loss = loss.forward(y_hat, y)
        loss_ner = loss_ner(y_hat_ner,batch["ner_list"].view(-1).to(torch.int64))
        loss = loss + loss_ner
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
        y_hat_ner = self(x).ner_logits
        loss = F.cross_entropy(y_hat, y)
        #logits_ner.view(-1, 13), ner_tensor_list.view(-1).to(torch.int64)
        loss_ner = F.cross_entropy(y_hat_ner,x["ner_list"].view(-1).to(torch.int64))
        loss = loss + loss_ner
        pred = y_hat.argmax(dim=1)
        correct = pred.eq(y.view_as(pred)).sum().item()
        micro_f1 = klue_re_micro_f1(pred.detach().cpu(), y.detach().cpu()).item()

        preds = {"val_micro_f1": micro_f1, "val_loss" : loss, "correct" : correct}
        self.validation_step_outputs.append(preds)

        self.validation_preds.extend(pred.tolist())
        self.validation_labels.extend(y.tolist())
        return preds

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        # val_micro_f1 = statistics.mean([x['val_micro_f1'] for x in self.validation_step_outputs])
        val_preds = torch.tensor(self.validation_preds).detach().cpu()
        val_labels = torch.tensor(self.validation_labels).detach().cpu()
        val_micro_f1 = klue_re_micro_f1(val_preds, val_labels)

        self.log_dict({'val_micro_f1': val_micro_f1, 'val_loss': avg_loss})

        if self.current_epoch >= 0:
            print(f"{{Epoch {self.current_epoch} val_micro_f1': {val_micro_f1} val_loss : {avg_loss}}}")
            show_confusion_matrix(preds=val_preds, labels=val_labels, epoch=self.current_epoch, save_path=self.confusion_matrix_path)

        self.validation_step_outputs.clear()
        self.validation_preds.clear()
        self.validation_labels.clear()

    def test_step(self, batch, _):
        x = batch
        y_hat = self(x).logits
        prob = F.softmax(y_hat, dim=-1).detach().cpu().numpy()
        pred = y_hat.argmax(dim=1, keepdim=True)
        self.output_pred.extend(pred.squeeze(1).tolist())
        self.output_prob.extend(prob.tolist())
        return pred

    def on_test_epoch_end(self):
        pred_answer = num_to_label(self.output_pred)
        test_id = list(range(len(self.output_pred)))
        output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':self.output_prob})
        os.makedirs("prediction", exist_ok=True)
        output.to_csv('./prediction/submission.csv', index=False)
