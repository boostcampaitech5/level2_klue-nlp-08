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

    dataset_dir = os.path.join(prj_dir, os.pardir, "dataset", "test", "test_data.csv")
    dataloader = ERDataModule(dataset_dir=dataset_dir, tokenizer=tokenizer, batch_size=16)
    model = ERNet.load_from_checkpoint("/opt/ml/level2_klue-nlp-08/model/klue/bert-base/2023-05-08 14:47:42.317246/epoch=2-step=4872.ckpt", learning_rate=0.00005, weight_decay=0.01)
    trainer = pl.Trainer(max_epochs = 3)
    trainer.test(model=model, datamodule=dataloader)
