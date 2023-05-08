import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from dataloader import ERDataModule
from model import ERNet

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    prj_dir = os.path.dirname(os.path.abspath(__file__))
    
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset_dir = os.path.join(prj_dir, os.pardir, "dataset", "train", "train.csv")
    dataloader = ERDataModule(dataset_dir=dataset_dir, tokenizer=tokenizer, batch_size=16)
    model = ERNet(model_name=MODEL_NAME, learning_rate=0.00005, weight_decay=0.01)

    trainer = pl.Trainer(callbacks=ModelCheckpoint(dirpath=f"./model/{MODEL_NAME.replace('/', '_')}/{datetime.now()}/", filename="{epoch}-{val_micro_f1}", monitor="val_micro_f1"), max_epochs = 3)
    trainer.fit(model = model, train_dataloaders=dataloader)

