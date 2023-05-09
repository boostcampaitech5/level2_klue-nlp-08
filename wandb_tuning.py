import os
from datetime import datetime

import pytorch_lightning as pl
import pytz
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

import wandb
from dataloader import ERDataModule
from model import ERNet

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    prj_dir = os.path.dirname(os.path.abspath(__file__))
    
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    parameters_dict = {
        'epochs': {
            'values': [5, 10]  
        },
        'batch_size': {
            'values': [4, 8, 16]  
        },
        'learning_rate': {
            'values': [0.00002, 0.00003, 0.00005]
        },
        'weight_decay': {
            'values': [0.0, 0.01]
        },
    }

    sweep_config = {
        'method': 'random',
        'parameters': parameters_dict
    }

    now = datetime.now(pytz.timezone("Asia/Seoul"))
    sweep_id = wandb.sweep(sweep_config, project=f"{MODEL_NAME.replace('/', '_')}_{now.strftime('%Y-%m-%d %H.%M.%S')}")

    def wandb_tuning():
        wandb.init()
        dataset_dir = os.path.join(prj_dir, os.pardir, "dataset", "train", "train.csv")
        dataloader = ERDataModule(dataset_dir=dataset_dir, tokenizer=tokenizer, batch_size=wandb.config.batch_size)
        model = ERNet(model_name=MODEL_NAME, learning_rate=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)

        wandb_logger = WandbLogger()
        trainer = pl.Trainer(callbacks=ModelCheckpoint(dirpath=f"./checkpoint/{MODEL_NAME.replace('/', '_')}/{now.strftime('%Y-%m-%d %H:%M:%S')}/", filename="{epoch}-{val_micro_f1}", monitor="val_micro_f1"), max_epochs = wandb.config.epochs, logger=wandb_logger)
        trainer.fit(model = model, train_dataloaders=dataloader)

    wandb.agent(sweep_id=sweep_id, function=wandb_tuning, count=1)
