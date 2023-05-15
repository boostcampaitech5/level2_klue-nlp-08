import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


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

            self.train_data = self.make_dataset(dataset_type=self.dataset_type, df=df_train, tokenizer=self.tokenizer, tokenizer_max_len=self.tokenizer_max_len, state="train")
            self.val_data = self.make_dataset(dataset_type=self.dataset_type, df=df_val, tokenizer=self.tokenizer, tokenizer_max_len=self.tokenizer_max_len, state="val")

        elif stage == "test":
            df_test = pd.read_csv(self.test_dataset_dir)
            
            self.test_data = self.make_dataset(dataset_type=self.dataset_type, df=df_test, tokenizer=self.tokenizer, tokenizer_max_len=self.tokenizer_max_len, state="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle = True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.val_batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.test_batch_size)

    def preprocessing_dataset(self, dataset : pd.DataFrame) -> pd.DataFrame:
        """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        subject_entity = []
        subject_s_idx = []
        subject_e_idx = []
        object_entity = []
        object_s_idx = []
        object_e_idx = []
        for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
            sub_dict, obj_dict = eval(i), eval(j)
            subject = sub_dict["word"]
            sub_s_idx = sub_dict["start_idx"]
            sub_e_idx = sub_dict["end_idx"]
            object = obj_dict["word"]
            obj_s_idx = obj_dict["start_idx"]
            obj_e_idx = obj_dict["end_idx"]

            subject_entity.append(subject)
            subject_s_idx.append(sub_s_idx)
            subject_e_idx.append(sub_e_idx)
            object_entity.append(object)
            object_s_idx.append(obj_s_idx)
            object_e_idx.append(obj_e_idx)
        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'subject_start_idx':subject_s_idx,'subject_end_idx':subject_e_idx,'object_entity':object_entity,'object_start_idx': object_s_idx,'object_end_idx':object_e_idx,'label':dataset['label'],})
        return out_dataset

    def tokenized_dataset(self, dataset : pd.DataFrame, tokenizer: AutoTokenizer) -> Dict:
        """ tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = []
        for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
            temp = e01 + '[SEP]' + e02
            concat_entity.append(temp)
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer_max_len,
            add_special_tokens=True,
            )
        return tokenized_sentences

    def tokenized_dataset_with_special_token(self, dataset : pd.DataFrame, tokenizer: AutoTokenizer, special_tokens: List) -> Dict:
        """ special token이 추가된 tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = []
        for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
            temp = e01 + '[SEP]' + e02
            concat_entity.append(temp)
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer_max_len,
            add_special_tokens=True,
            )
        return tokenized_sentences

    def label_to_num(self, label : str) -> int:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"pickle", 'dict_label_to_num.pkl'), 'rb') as f:
            dict_label_to_num = pickle.load(f)
        return dict_label_to_num[label]

    def make_dataset(self, df : pd.DataFrame, state : str) -> List[Dict]:
        result = []
        df_preprocessed = self.preprocessing_dataset(dataset = df)
        df_tokenized = self.tokenized_dataset(dataset=df_preprocessed, tokenizer=self.tokenizer)
        for idx in range(len(df_tokenized["input_ids"])):
            temp = {key: val[idx] for key, val in df_tokenized.items()}
            if state != "test":
                temp['labels'] = self.label_to_num(df['label'].values[idx])
            result.append(temp)

        return result
