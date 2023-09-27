from typing import List

import pandas as pd
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


class TaptDataSet(Dataset):
    def __init__(self, file_path_list: List, tokenizer: PreTrainedTokenizer):

        for i in range(len(file_path_list)):
            df = pd.read_csv(file_path_list[i])
            self.sentence_list = []
            for i in range(len(df)):
                sent = df.loc[i, "sentence"]
                self.sentence_list.append(sent)

        self.tokenizer = tokenizer

        self.tokenized_sentences = []

        self.tokenized_sent = self.tokenizer(
            self.sentence_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )

        for idx in range(len(self.tokenized_sent["input_ids"])):
            temp = {key: val[idx] for key, val in self.tokenized_sent.items()}
            self.tokenized_sentences.append(temp)

    def __len__(self):
        return len(self.tokenized_sentences)

    def __getitem__(self, idx):
        return self.tokenized_sentences[idx]
