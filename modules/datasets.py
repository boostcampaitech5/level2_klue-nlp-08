from typing import Dict, List

import pandas as pd
from modules.preprocess import preprocessing_dataset
from modules.tokenize import (
    tokenized_dataset,
    tokenized_dataset_ainize_token,
    tokenized_dataset_type_entity_token,
    tokenized_dataset_type_entity_token_v2,
    tokenized_dataset_type_punct_token,
)
from modules.utils import label_to_num, make_index_ids_roberta, make_ner
from transformers import AutoTokenizer


def make_dataset(
    dataset_type: str,
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    tokenizer_max_len: int,
    state: str,
) -> List[Dict]:
    if dataset_type == "default":
        dataset = make_dataset_for_roberta
    elif dataset_type == "punct":
        dataset = make_dataset_punct
    elif dataset_type == "type_entity":
        dataset = RE_make_dataset_for_roberta_attention
    elif dataset_type == "ainize":
        dataset = make_dataset_for_ainize
    elif dataset_type == "type_entity_v2":
        dataset = make_dataset_with_type_entity_token
    else:
        raise ValueError("정의되지 않은 dataset type입니다.")

    return dataset(
        df=df, tokenizer=tokenizer, tokenizer_max_len=tokenizer_max_len, state=state
    )


def make_dataset_for_roberta(
    df: pd.DataFrame, tokenizer: AutoTokenizer, tokenizer_max_len: int, state: str
) -> List[Dict]:
    result = []
    df_preprocessed = preprocessing_dataset(dataset=df)
    df_tokenized = tokenized_dataset(
        dataset=df_preprocessed,
        tokenizer=tokenizer,
        tokenizer_max_len=tokenizer_max_len,
    )

    for idx in range(len(df_tokenized["input_ids"])):
        temp = {key: val[idx] for key, val in df_tokenized.items()}
        if state != "test":
            temp["labels"] = label_to_num(df["label"].values[idx])
        result.append(temp)

    return result


def make_dataset_punct(
    df: pd.DataFrame, tokenizer: AutoTokenizer, tokenizer_max_len: int, state: str
) -> List[Dict]:
    result = []
    df_preprocessed = preprocessing_dataset(dataset=df)
    df_tokenized = tokenized_dataset_type_punct_token(
        dataset=df_preprocessed,
        tokenizer=tokenizer,
        tokenizer_max_len=tokenizer_max_len,
    )

    for idx in range(len(df_tokenized["input_ids"])):
        temp = {key: val[idx] for key, val in df_tokenized.items()}
        if state != "test":
            temp["labels"] = label_to_num(df["label"].values[idx])
        result.append(temp)

    return result


def RE_make_dataset_for_roberta_attention(
    df: pd.DataFrame, tokenizer: AutoTokenizer, tokenizer_max_len: int, state: str
) -> List[Dict]:
    result = []

    df_preprocessed = preprocessing_dataset(dataset=df)
    df_tokenized = tokenized_dataset_type_entity_token(
        dataset=df_preprocessed,
        tokenizer=tokenizer,
        tokenizer_max_len=tokenizer_max_len,
    )
    ner_list = make_ner(df_tokenized)
    index_ids = make_index_ids_roberta(tokenized_sentences=df_tokenized)

    df_tokenized.data["index_ids"] = index_ids
    df_tokenized.data["ner_list"] = ner_list

    for idx in range(len(df_tokenized["input_ids"])):
        temp = {key: val[idx] for key, val in df_tokenized.items()}
        if state != "test":
            temp["labels"] = label_to_num(df["label"].values[idx])
        result.append(temp)
    return result


def make_dataset_for_ainize(
    df: pd.DataFrame, tokenizer: AutoTokenizer, tokenizer_max_len: int, state: str
) -> List[Dict]:
    result = []
    df_preprocessed = preprocessing_dataset(dataset=df)
    df_tokenized = tokenized_dataset_ainize_token(
        dataset=df_preprocessed,
        tokenizer=tokenizer,
        tokenizer_max_len=tokenizer_max_len,
    )

    for idx in range(len(df_tokenized["input_ids"])):
        temp = {key: val[idx] for key, val in df_tokenized.items()}
        if state != "test":
            temp["labels"] = label_to_num(df["label"].values[idx])
        result.append(temp)

    return result


def make_dataset_with_type_entity_token(
    df: pd.DataFrame, tokenizer: AutoTokenizer, tokenizer_max_len: int, state: str
) -> List[Dict]:
    result = []
    df_preprocessed = preprocessing_dataset(dataset=df)
    df_tokenized = tokenized_dataset_type_entity_token_v2(
        dataset=df_preprocessed,
        tokenizer=tokenizer,
        tokenizer_max_len=tokenizer_max_len,
    )

    for idx in range(len(df_tokenized["input_ids"])):
        temp = {key: val[idx] for key, val in df_tokenized.items()}
        if state != "test":
            temp["labels"] = label_to_num(df["label"].values[idx])
        result.append(temp)

    return result
