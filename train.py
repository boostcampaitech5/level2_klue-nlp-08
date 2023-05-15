import os
from datetime import datetime

import pytorch_lightning as pl
import pytz
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from dataloader import ERDataModule
from model import ERNet
from utils import config_parser

if __name__ == "__main__":
    config = config_parser()

    pl.seed_everything(config["seed"], workers=True)

    prj_dir = os.path.dirname(os.path.abspath(__file__))

    MODEL_NAME = config["model"]["model_name"]
    additional_special_tokens_list = [
        "[SUB:LOC]",
        "[SUB:PER]",
        "[SUB:ORG]",
        "[OBJ:NOH]",
        "[OBJ:DAT]",
        "[OBJ:PER]",
        "[OBJ:POH]",
        "[OBJ:LOC]",
        "[OBJ:ORG]",
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, additional_special_tokens=additional_special_tokens_list
    )

    dataset_dir = os.path.join(prj_dir, os.pardir, "dataset", "train", "train.csv")
    dataloader = ERDataModule(config=config, tokenizer=tokenizer)
    model = ERNet(config=config)

    now = datetime.now(pytz.timezone("Asia/Seoul"))

    trainer = pl.Trainer(
        callbacks=ModelCheckpoint(
            dirpath=f"./checkpoint/{config['model']['model_name'].replace('/', '_')}/{now.strftime('%Y-%m-%d %H.%M.%S')}/",
            filename="{epoch}-{val_micro_f1:.2f}",
            monitor="val_micro_f1",
            mode="max",
        ),
        max_epochs=config["train"]["num_train_epoch"],
    )
    trainer.fit(model=model, train_dataloaders=dataloader)
