import os

import pytorch_lightning as pl
import torch
from dataloader import ERDataModule
from model import ERNet
from modules.utils import config_parser, get_special_token
from transformers import AutoTokenizer

if __name__ == "__main__":
    config = config_parser()

    pl.seed_everything(config["seed"], workers=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prj_dir = os.path.dirname(os.path.abspath(__file__))

    MODEL_NAME = config["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        'klue/roberta-large',
        additional_special_tokens=get_special_token(config["train"]["dataset_type"]),
    )

    dataloader = ERDataModule(config=config, tokenizer=tokenizer)
    model = ERNet.load_from_checkpoint(
        config["path"]["model"], config=config, resize_token_embedding=len(tokenizer)
    )
    trainer = pl.Trainer(max_epochs=config["train"]["num_train_epoch"])
    trainer.test(model=model, datamodule=dataloader)
