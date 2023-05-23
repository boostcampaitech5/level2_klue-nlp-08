from datetime import datetime

import pytorch_lightning as pl
import pytz
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

import wandb
from dataloader import ERDataModule
from model import ERNet
from modules.utils import config_parser

if __name__ == "__main__":
    config = config_parser()

    pl.seed_everything(config["seed"], workers=True)


    MODEL_NAME = config["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    parameters = OmegaConf.to_container(config["wandb"]["parameters"])

    sweep_config = {
        "name": config["wandb"]["name"],
        "method": config["wandb"]["method"],
        "parameters": parameters,
    }

    now = datetime.now(pytz.timezone("Asia/Seoul"))
    sweep_id = wandb.sweep(
        sweep_config, project=f"{config['wandb']['project_name']}", entity="Yoonseul"
    )

    def wandb_tuning():
        wandb.init()
        dataloader = ERDataModule(
            config=config, tokenizer=tokenizer, wandb_batch_size=wandb.config.batch_size
        )
        model = ERNet(config=config, wandb_config=wandb.config, state="train")

        wandb_logger = WandbLogger()
        trainer = pl.Trainer(
            callbacks=ModelCheckpoint(
                dirpath=f"./checkpoint/{MODEL_NAME.replace('/', '_')}/{now.strftime('%Y-%m-%d %H.%M.%S')}/",
                filename="{epoch}-{val_micro_f1:.2f}",
                monitor="val_micro_f1",
                mode="max",
            ),
            max_epochs=wandb.config.epochs,
            logger=wandb_logger,
        )
        trainer.fit(model=model, train_dataloaders=dataloader)

    wandb.agent(sweep_id=sweep_id, function=wandb_tuning, count=36)
