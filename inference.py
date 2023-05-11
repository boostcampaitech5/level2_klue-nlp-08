import os

import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer

from dataloader import ERDataModule
from model import ERNet
from model_list import TAEMIN_TOKEN_ATTENTION_RoBERTa
from utils import config_parser

if __name__ == "__main__":

    config = config_parser()

    pl.seed_everything(config["seed"], workers=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    prj_dir = os.path.dirname(os.path.abspath(__file__))

    MODEL_NAME = config["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    added_token_num = tokenizer.add_special_tokens({"additional_special_tokens": ["[ORG]", "[PER]", "[LOC]", "[POH]",
                                                                                  "[DAT]", "[NOH]", "[/ORG]", "[/PER]",
                                                                                  "[/LOC]", "[/POH]", "[/DAT]",
                                                                                  "[/NOH]", "[SUB]", "[/SUB]", "[OBJ]",
                                                                                  "[/OBJ]", ]})
    dataset_dir = os.path.join(prj_dir, os.pardir, "dataset", "test", "test.csv")
    dataloader = ERDataModule(config=config, tokenizer=tokenizer)
    model = ERNet(config=config).load_from_checkpoint('C:/Users/tm011/PycharmProjects/level2_klue-nlp-08/checkpoint/klue_roberta-large/2023-05-11 22.17.06/epoch=2-val_micro_f1=70.19.ckpt',config=config)
    trainer = pl.Trainer(max_epochs = config["train"]["num_train_epoch"])
    trainer.test(model=model, datamodule=dataloader)
