import os
from datetime import datetime

import pytorch_lightning as pl
import pytz
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from dataloader import ERDataModule
from model import ERNet
from modules.utils import config_parser, get_special_token

if __name__ == "__main__":
    config = config_parser()

    pl.seed_everything(config["seed"], workers=True)

    prj_dir = os.path.dirname(os.path.abspath(__file__))

    MODEL_NAME = config["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        additional_special_token=get_special_token(config["train"]["dataset_type"]),
    )

    dataloader = ERDataModule(config=config, tokenizer=tokenizer)
    model = ERNet(config=config, state="train")
    model.model.resize_token_embeddings(len(tokenizer))

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
