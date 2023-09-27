import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pandas as pd
from torch.utils.data.dataset import Dataset


class AugmentDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.file_path_list = [
            config["data_path"]["train_path"],
            config["data_path"]["test_path"],
        ]
        for i in range(len(self.file_path_list)):
            df = pd.read_csv(self.file_path_list[i])
            self.sentence_list = []
            for i in range(len(df)):
                sent = df.loc[i, "sentence"]
                self.sentence_list.append(sent)

        self.tokenizer = tokenizer

        self.tokenized_sentences = []

        self.tokenized_sentence = self.tokenizer(
            self.sentence_list,
            return_tensors=config["tokenizers"]["return_tensors"],
            padding=config["tokenizers"]["padding"],
            truncation=config["tokenizers"]["truncation"],
            max_length=config["tokenizers"]["maxlength"],
            add_special_tokens=config["tokenizers"]["add_special_tokens"],
        )

        for idx in range(len(self.tokenized_sentence["input_ids"])):
            temp = {key: val[idx] for key, val in self.tokenized_sentence.items()}
            self.tokenized_sentences.append(temp)

    def __len__(self):
        return len(self.tokenized_sentences)

    def __getitem__(self, idx):
        return self.tokenized_sentences[idx]
