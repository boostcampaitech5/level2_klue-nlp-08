import os
import re

import torch

import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding


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

    def setup(self, stage: str):
        if stage == "fit":
            df_train, df_val = pd.read_csv(self.train_dataset_dir), pd.read_csv(self.dev_dataset_dir)
            self.train_data = self.RE_make_dataset(df_train, state = "train")
            self.val_data = self.RE_make_dataset(df_val, state = "val")
            # df_train, df_val = [self.preprocessing_dataset(dataset = df_train), self.preprocessing_dataset(dataset = df_val)]
            # train_tokenized, val_tokenized = [self.tokenized_dataset(dataset=df_train, tokenizer=self.tokenizer), self.tokenized_dataset(dataset=df_val, tokenizer=self.tokenizer)]
            # self.train_data, self.val_data = [train_tokenized, self.tokenized_dataset(dataset=df_val, tokenizer=self.tokenizer)]
            # self.train_data['labels'] = self.label_to_num(df_train['label'].values)
            # self.val_data['labels'] = self.label_to_num(df_val['label'].values)

        elif stage == "test":
            df_test = pd.read_csv(self.test_dataset_dir)
            self.test_data = self.make_dataset(df_test, state = "test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle = True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.val_batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.test_batch_size)

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

    def tokenized_dataset(self, dataset : pd.DataFrame, tokenizer: AutoTokenizer) -> Dict:
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

    def RE_make_dataset(self, df : pd.DataFrame, state : str) -> List[Dict]:
        result = []
        df_preprocessed = self.preprocessing_dataset_sub_obj_entity(dataset = df)
        df_tokenized = self.tokenized_dataset_type_entity_token(dataset=df_preprocessed, tokenizer=self.tokenizer)
        index_ids = self.make_index_ids(tokenized_sentences=df_tokenized)

        df_tokenized.data['index_ids'] = index_ids
        for idx in range(len(df_tokenized["input_ids"])):
            temp = {key: val[idx] for key, val in df_tokenized.items()}
            if state != "test":
                temp['labels'] = self.label_to_num(df['label'].values[idx])
            result.append(temp)

        return result

    def preprocessing_dataset_sub_obj_entity(self, dataset: pd.DataFrame) -> pd.DataFrame:
        subject_entity = []
        object_entity = []
        subject_entity_idx = []
        object_entity_idx = []
        subject_entity_type = []
        object_entity_type = []
        english_pattern = re.compile(r"[^a-zA-Z\s]")
        for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
            s = i[1:-1].split(',')[0].split(':')[1]
            o = j[1:-1].split(',')[0].split(':')[1]

            s_start_idx = int(re.sub("[^0-9]", "", (i.split('start_idx')[1][3:15])))
            s_end_idx = int(re.sub("[^0-9]", "", (i.split('start_idx')[1][15:])))

            o_start_idx = int(re.sub("[^0-9]", "", (j.split('start_idx')[1][3:15])))
            o_end_idx = int(re.sub("[^0-9]", "", (j.split('start_idx')[1][15:])))

            subject_entity_type_word = i.split("'type': ")[-1]
            object_entity_type_word = j.split("'type': ")[-1]

            subject_entity_type_word = re.sub(english_pattern, "", subject_entity_type_word)
            object_entity_type_word = re.sub(english_pattern, "", object_entity_type_word)

            subject_entity.append(s)
            object_entity.append(o)
            subject_entity_idx.append((s_start_idx, s_end_idx))
            object_entity_idx.append((o_start_idx, o_end_idx))
            subject_entity_type.append(subject_entity_type_word)
            object_entity_type.append(object_entity_type_word)

        out_dataset = pd.DataFrame(
            {'id': dataset['id'], 'sentence': dataset['sentence'], 'subject_entity': subject_entity,
             'object_entity': object_entity, 'label': dataset['label'],
             'subject_entity_idx': subject_entity_idx, 'object_entity_idx': object_entity_idx,
             'subject_entity_type': subject_entity_type, 'object_entity_type': object_entity_type})
        return out_dataset
    def tokenized_dataset_type_entity_token(self, dataset: pd.DataFrame, tokenizer: AutoTokenizer) -> Dict:
        concat_entity = []
        sentence_list = []
        for e01, e02, sentence, s_e_i, o_e_i, s_type_str, o_type_str in zip(dataset['subject_entity'],
                                                                            dataset['object_entity'],
                                                                            dataset['sentence'],
                                                                            dataset['subject_entity_idx'],
                                                                            dataset['object_entity_idx']
                , dataset['subject_entity_type'], dataset['object_entity_type']):

            temp = ''
            e01 = re.sub(r'^\s+', '', e01)  # remove leading spaces
            e01 = re.sub(r"'", '', e01)  # remove apostrophes
            e02 = re.sub(r'^\s+', '', e02)  # remove leading spaces
            e02 = re.sub(r"'", '', e02)

            s_type = f'[{s_type_str}]'
            end_s_type = f'[/{s_type_str}]'

            o_type = f'[{o_type_str}]'
            end_o_type = f'[/{o_type_str}]'

            temp = s_type + e01 + end_s_type + '[SEP]' + o_type + e02 + end_o_type
            concat_entity.append(temp)  # '[Entity]'
            if s_e_i[0] < o_e_i[0]:
                sentence = sentence[:s_e_i[0]] + s_type + sentence[s_e_i[0]:]
                sentence = sentence[:s_e_i[1] + 6] + end_s_type + sentence[s_e_i[1] + 6:]
                sentence = sentence[:o_e_i[0] + 11] + o_type + sentence[o_e_i[0] + 11:]  # 8
                sentence = sentence[:o_e_i[1] + 11 + 6] + end_o_type + sentence[o_e_i[1] + 11 + 6:]  # 9
            else:
                sentence = sentence[:o_e_i[0]] + o_type + sentence[o_e_i[0]:]
                sentence = sentence[:o_e_i[1] + 6] + end_o_type + sentence[o_e_i[1] + 6:]
                sentence = sentence[:s_e_i[0] + 11] + s_type + sentence[s_e_i[0] + 11:]  # 8
                sentence = sentence[:s_e_i[1] + 11 + 6] + end_s_type + sentence[s_e_i[1] + 11 + 6:]  # 9
            sentence_list.append(sentence)
        tokenized_sentences = tokenizer(
            concat_entity,
            sentence_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
        return tokenized_sentences

    def make_index_ids(self, tokenized_sentences: Dict) -> BatchEncoding:
        index_ids = []
        for i in range(len(tokenized_sentences.data['attention_mask'])):
            index_number = 0
            temp_input_ids = tokenized_sentences.data['input_ids'][i]
            index_tensor = torch.zeros_like(temp_input_ids)

            for index, value in enumerate(temp_input_ids):
                if value == 0:
                    break
                if value >= 32000 and value <= 32005:
                    index_number = 1
                if value > 32005:
                    index_number = 0
                index_tensor[index] = index_number
                if value >= 32000 and value <= 32005:
                    index_tensor[index] = 0
            index_ids.append(list(index_tensor.to(dtype=torch.int64)))
        return torch.tensor(index_ids)