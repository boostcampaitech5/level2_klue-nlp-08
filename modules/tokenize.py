from typing import Dict

import pandas as pd
from transformers import AutoTokenizer

type_dict = {'ORG': '단체','PER':'사람','LOC':'지역','POH':'직업','NOH':'숫자','DAT':'날짜'}

# def tokenized_dataset(dataset : pd.DataFrame, tokenizer: AutoTokenizer, tokenizer_max_len : int) -> Dict:
#     """예시 : 비틀즈[SEP]조지 해리슨[SEP]〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다."""
#     concat_entity = []
#     for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
#         temp = e01 + '[SEP]' + e02
#         concat_entity.append(temp)

#     tokenized_sentences = tokenizer(
#         concat_entity,
#         list(dataset['sentence']),
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=tokenizer_max_len,
#         add_special_tokens=True,
#         )
#     return tokenized_sentences

def tokenized_dataset(dataset : pd.DataFrame, tokenizer: AutoTokenizer, tokenizer_max_len : int) -> Dict:
    """예시 : [ORG]비틀즈[/ORG][SEP][PER]조지해리슨[/PER]〈Something〉는 [PER]조지 해리슨[/PER]이 쓰고 [ORG]비틀즈[ORG]가 1969년 앨범 《Abbey Road》에 담은 노래다. """
    concat_entity = []
    for e01, sub_type, e02, obj_type in zip(dataset['subject_entity'], dataset["subject_entity_type"], dataset['object_entity'], dataset["object_entity_type"]):
        s_type = f'[{sub_type}]' #사람
        end_s_type = f'[/{sub_type}]'

        o_type = f'[{obj_type}]'
        end_o_type = f'[/{obj_type}]'

        temp = s_type + e01 + end_s_type + '[SEP]' + o_type + e02 + end_o_type
        concat_entity.append(temp)

    new_sentences = []
    for sent, sub_s, sub_e, sub_type, obj_s, obj_e, obj_type in zip(dataset["sentence"], dataset["subject_start_idx"], dataset["subject_end_idx"], dataset["subject_entity_type"], dataset["object_start_idx"], dataset["object_end_idx"], dataset["object_entity_type"]):

        if sub_e < obj_e:
            new_sent = sent[:sub_s] + s_type + sent[sub_s:sub_e+1] + end_s_type + sent[sub_e+1:obj_s] + o_type + sent[obj_s:obj_e+1] + end_o_type + sent[obj_e+1:]
        else:
            new_sent = sent[:obj_s] + o_type + sent[obj_s:obj_e+1] + end_o_type + sent[obj_e+1:sub_s] + s_type + sent[sub_s:sub_e+1] + end_s_type + sent[sub_e+1:]
        new_sentences.append(new_sent)

    tokenized_sentences = tokenizer(
        concat_entity,
        new_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer_max_len,
        add_special_tokens=True,
    )

    return tokenized_sentences

def tokenized_dataset_type_punct_token(dataset : pd.DataFrame, tokenizer: AutoTokenizer, tokenizer_max_len : int) -> Dict:
    """예시 : @*단체*비틀즈@[REL]#*사람*조지 해리슨#[SEP] 〈Something〉는 #*사람*조지 해리슨#이 쓰고 @*단체*비틀즈@가 1969년 앨범 《Abbey Road》에 담은 노래다."""
    concat_entity = []
    for e01, sub_type, e02, obj_type in zip(dataset['subject_entity'], dataset["subject_entity_type"], dataset['object_entity'], dataset["object_entity_type"]):
        s_type = f'{type_dict[sub_type]}' #사람

        o_type = f'{type_dict[obj_type]}'

        temp = '@' +'*'+s_type+ '*' + e01 + '@' + '[REL]' + '#'+'*'+o_type+ '*' + e02 + '#'
        concat_entity.append(temp)

    new_sentences = []
    for sent, sub_s, sub_e, sub_type, obj_s, obj_e, obj_type in zip(dataset["sentence"], dataset["subject_start_idx"], dataset["subject_end_idx"], dataset["subject_entity_type"], dataset["object_start_idx"], dataset["object_end_idx"], dataset["object_entity_type"]):
        if sub_e < obj_e:
            new_sent = sent[:sub_s] + f"@*{s_type}*" + sent[sub_s:sub_e+1] + "@" + sent[sub_e+1:obj_s] + f"#*{o_type}*" + sent[obj_s:obj_e+1] + "#" + sent[obj_e+1:]
        else:
            new_sent = sent[:obj_s] + f"#*{o_type}*" + sent[obj_s:obj_e+1] + "#" + sent[obj_e+1:sub_s] + f"@*{s_type}*" + sent[sub_s:sub_e+1] + "@" + sent[sub_e+1:]
        new_sentences.append(new_sent)

    tokenized_sentences = tokenizer(
        concat_entity,
        new_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer_max_len,
        add_special_tokens=True,
    )

    return tokenized_sentences

def tokenized_dataset_type_entity_token(dataset : pd.DataFrame, tokenizer: AutoTokenizer, tokenizer_max_len : int) -> Dict:
    """예시 : [ORG]비틀즈[/ORG][SEP][PER]조지해리슨[/PER]〈Something〉는 [PER]조지 해리슨[/PER]이 쓰고 [ORG]비틀즈[ORG]가 1969년 앨범 《Abbey Road》에 담은 노래다. """
    concat_entity = []
    for e01, sub_type, e02, obj_type in zip(dataset['subject_entity'], dataset["subject_entity_type"], dataset['object_entity'], dataset["object_entity_type"]):
        s_type = f'[{sub_type}]' #사람
        end_s_type = f'[/{sub_type}]'

        o_type = f'[{obj_type}]'
        end_o_type = f'[/{obj_type}]'

        temp = s_type + e01 + end_s_type + '[SEP]' + o_type + e02 + end_o_type
        concat_entity.append(temp)

    new_sentences = []
    for sent, sub_s, sub_e, sub_type, obj_s, obj_e, obj_type in zip(dataset["sentence"], dataset["subject_start_idx"], dataset["subject_end_idx"], dataset["subject_entity_type"], dataset["object_start_idx"], dataset["object_end_idx"], dataset["object_entity_type"]):

        if sub_e < obj_e:
            new_sent = sent[:sub_s] + s_type + sent[sub_s:sub_e+1] + end_s_type + sent[sub_e+1:obj_s] + o_type + sent[obj_s:obj_e+1] + end_o_type + sent[obj_e+1:]
        else:
            new_sent = sent[:obj_s] + o_type + sent[obj_s:obj_e+1] + end_o_type + sent[obj_e+1:sub_s] + s_type + sent[sub_s:sub_e+1] + end_s_type + sent[sub_e+1:]
        new_sentences.append(new_sent)

    tokenized_sentences = tokenizer(
        concat_entity,
        new_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer_max_len,
        add_special_tokens=True,
    )

    return tokenized_sentences

def tokenized_dataset_ainize_token(dataset : pd.DataFrame, tokenizer: AutoTokenizer, tokenizer_max_len : int) -> Dict:
    """예시 : <subj>비틀즈</subj><obj>조지해리슨</obj>〈Something〉는 <subj>조지 해리슨</subj>이 쓰고 <obj>비틀즈</obj>가 1969년 앨범 《Abbey Road》에 담은 노래다. """
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        s_type = "<subj>" #사람
        end_s_type = "</subj>"

        o_type = "<obj>"
        end_o_type = "</obj>"

        temp = s_type + e01 + end_s_type + o_type + e02 + end_o_type
        concat_entity.append(temp)


    new_sentences = []
    for sent, sub_s, sub_e, obj_s, obj_e in zip(dataset["sentence"], dataset["subject_start_idx"], dataset["subject_end_idx"], dataset["object_start_idx"], dataset["object_end_idx"]):

        if sub_e < obj_e:
            new_sent = sent[:sub_s] + s_type + sent[sub_s:sub_e+1] + end_s_type + sent[sub_e+1:obj_s] + o_type + sent[obj_s:obj_e+1] + end_o_type + sent[obj_e+1:]
        else:
            new_sent = sent[:obj_s] + o_type + sent[obj_s:obj_e+1] + end_o_type + sent[obj_e+1:sub_s] + s_type + sent[sub_s:sub_e+1] + end_s_type + sent[sub_e+1:]
        new_sentences.append(new_sent)

    tokenized_sentences = tokenizer(
        concat_entity,
        new_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer_max_len,
        add_special_tokens=True,
    )

    return tokenized_sentences


def tokenized_dataset_type_entity_token_v2(dataset : pd.DataFrame, tokenizer: AutoTokenizer, tokenizer_max_len : int) -> Dict:
    """예시 : [ORG]비틀즈[/ORG][SEP][PER]조지해리슨[/PER][SEP]〈Something〉는 [PER]조지 해리슨[/PER]이 쓰고 [ORG]비틀즈[ORG]가 1969년 앨범 《Abbey Road》에 담은 노래다. """
    concat_entity = []
    for e01, sub_type, e02, obj_type in zip(dataset['subject_entity'], dataset["subject_entity_type"], dataset['object_entity'], dataset["object_entity_type"]):
        s_type = f'[SUB:{sub_type}]' #사람
        end_s_type = f'[/SUB:{sub_type}]'

        o_type = f'[OBJ:{obj_type}]'
        end_o_type = f'[/OBJ:{obj_type}]'

        temp = s_type + e01 + end_s_type + '[SEP]' + o_type + e02 + end_o_type
        concat_entity.append(temp)

    new_sentences = []
    for sent, sub_s, sub_e, obj_s, obj_e in zip(dataset["sentence"], dataset["subject_start_idx"], dataset["subject_end_idx"], dataset["object_start_idx"], dataset["object_end_idx"]):

        if sub_e < obj_e:
            new_sent = sent[:sub_s] + s_type + sent[sub_s:sub_e+1] + end_s_type + sent[sub_e+1:obj_s] + o_type + sent[obj_s:obj_e+1] + end_o_type + sent[obj_e+1:]
        else:
            new_sent = sent[:obj_s] + o_type + sent[obj_s:obj_e+1] + end_o_type + sent[obj_e+1:sub_s] + s_type + sent[sub_s:sub_e+1] + end_s_type + sent[sub_e+1:]
        new_sentences.append(new_sent)

    tokenized_sentences = tokenizer(
        concat_entity,
        new_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer_max_len,
        add_special_tokens=True,
    )

    return tokenized_sentences