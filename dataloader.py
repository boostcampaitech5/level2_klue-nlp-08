import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from modules.datasets import make_dataset


class ERDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer : AutoTokenizer, wandb_batch_size : int =None):
        super().__init__()
        self.train_dataset_dir = config["path"]["train"]
        self.dev_dataset_dir = config["path"]["dev"]
        self.test_dataset_dir = config["path"]["test"]
        if wandb_batch_size:
            self.train_batch_size = wandb_batch_size
        else:
            self.train_batch_size = config["data"]["train_batch_size"]
        self.val_batch_size = config["data"]["val_batch_size"]
        self.test_batch_size = config["data"]["test_batch_size"]
        self.tokenizer = tokenizer
        self.tokenizer_max_len = config["data"]["tokenizer_max_len"]
        self.dataset_type = config["train"]["dataset_type"]

    def setup(self, stage: str):
        if stage == "fit":
            df_train, df_val = pd.read_csv(self.train_dataset_dir), pd.read_csv(self.dev_dataset_dir)

            self.train_data = make_dataset(dataset_type=self.dataset_type, df=df_train, tokenizer=self.tokenizer, tokenizer_max_len=self.tokenizer_max_len, state="train")
            self.val_data = make_dataset(dataset_type=self.dataset_type, df=df_val, tokenizer=self.tokenizer, tokenizer_max_len=self.tokenizer_max_len, state="val")

        elif stage == "test":
            df_test = pd.read_csv(self.test_dataset_dir)

            self.test_data = make_dataset(dataset_type=self.dataset_type, df=df_test, tokenizer=self.tokenizer, tokenizer_max_len=self.tokenizer_max_len, state="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle = True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.val_batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.test_batch_size)
