import os

import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer

from dataloader import ERDataModule
from model import ERNet
from utils import config_parser

if __name__ == "__main__":

    config = config_parser()

    pl.seed_everything(config["seed"], workers=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    prj_dir = os.path.dirname(os.path.abspath(__file__))

    MODEL_NAME = config["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,)

    # dataset_dir = os.path.join(prj_dir, os.pardir, "dataset", "test", "test_data.csv")
    dataloader = ERDataModule(config=config, tokenizer=tokenizer)
    model = ERNet.load_from_checkpoint(config["path"]["model"], config=config, resize_token_embedding=len(tokenizer))
    trainer = pl.Trainer(max_epochs = config["train"]["num_train_epoch"])
    trainer.test(model=model, datamodule=dataloader)
