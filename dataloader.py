import os
import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class ERDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir: str, tokenizer : AutoTokenizer, batch_size : int):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage: str):
        df_dataset = pd.read_csv(self.dataset_dir)

        if stage == "fit":
            df_train, df_val = df_dataset[:25976], df_dataset[25976:]
            self.train_data = self.make_dataset(df_train, state = "train")
            self.val_data = self.make_dataset(df_val, state = "val")
            # df_train, df_val = [self.preprocessing_dataset(dataset = df_train), self.preprocessing_dataset(dataset = df_val)]
            # train_tokenized, val_tokenized = [self.tokenized_dataset(dataset=df_train, tokenizer=self.tokenizer), self.tokenized_dataset(dataset=df_val, tokenizer=self.tokenizer)]
            # self.train_data, self.val_data = [train_tokenized, self.tokenized_dataset(dataset=df_val, tokenizer=self.tokenizer)]
            # self.train_data['labels'] = self.label_to_num(df_train['label'].values)
            # self.val_data['labels'] = self.label_to_num(df_val['label'].values)

        elif stage == "test":
            self.test_data = self.make_dataset(df_dataset, state = "test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle = True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def preprocessing_dataset(self, dataset : pd.DataFrame) -> pd.DataFrame:
        """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        subject_entity = []
        object_entity = []
        for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1]
            j = j[1:-1].split(',')[0].split(':')[1]

            subject_entity.append(i)
            object_entity.append(j)
        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
        return out_dataset

    def tokenized_dataset(self, dataset : pd.DataFrame, tokenizer: AutoTokenizer) -> dict:
        """ tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = []
        for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
            temp = ''
            temp = e01 + '[SEP]' + e02
            concat_entity.append(temp)
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            )
        return tokenized_sentences

    def label_to_num(self, label : str) -> int:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"pickle", 'dict_label_to_num.pkl'), 'rb') as f:
            dict_label_to_num = pickle.load(f)
        return dict_label_to_num[label]

    def make_dataset(self, df : pd.DataFrame, state : str) -> dict:
        result = []
        df_preprocessed = self.preprocessing_dataset(dataset = df)
        df_tokenized = self.tokenized_dataset(dataset=df_preprocessed, tokenizer=self.tokenizer)
        for idx in range(len(df_tokenized["input_ids"])):
            temp = {key: val[idx] for key, val in df_tokenized.items()}
            if state != "test":
                temp['labels'] = self.label_to_num(df['label'].values[idx])
            result.append(temp)

        return result
