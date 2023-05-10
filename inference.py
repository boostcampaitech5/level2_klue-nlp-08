import os

import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer

from dataloader import ERDataModule
from model import ERNet

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    prj_dir = os.path.dirname(os.path.abspath(__file__))

    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset_train_dir = os.path.join(prj_dir, "dataset", "data", "train.csv")
    dataset_val_dir = os.path.join(prj_dir, "dataset", "data", "val_.csv")
    dataset_test_dir = os.path.join(prj_dir, "dataset", "data", "test.csv")
    dataloader = ERDataModule(dataset_train_dir=dataset_train_dir,dataset_val_dir=dataset_val_dir,dataset_test_dir=dataset_test_dir, tokenizer=tokenizer, batch_size=16)
    model = ERNet.load_from_checkpoint("./checkpoint/klue_bert-base/2023-05-10_02_17_16/epoch=2-val_micro_f1=65.01.ckpt", learning_rate=0.00005, weight_decay=0.01)
    trainer = pl.Trainer(max_epochs = 3)
    trainer.test(model=model, datamodule=dataloader)
