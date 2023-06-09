import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pandas as pd
import torch
import torch.nn.functional as F
from modules.utils import config_parser
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from utils import find_word_indices

label_list = [
    "no_relation",
    "org:top_members/employees",
    "org:members",
    "org:product",
    "per:title",
    "org:alternate_names",
    "per:employee_of",
    "per:parents",
    "org:dissolved",
    "per:schools_attended",
    "per:date_of_death",
    "per:date_of_birth",
    "per:place_of_birth",
    "per:place_of_death",
    "org:founded_by",
    "per:religion",
    "org:founded",
    "org:political/religious_affiliation",
    "org:member_of",
]

if __name__ == "__main__":
    config = config_parser()

    dataframe = pd.read_csv(config["data_path"]["train_path"])

    new_index_ids = []
    new_sentences = []
    new_subject_entities = []
    new_object_entities = []
    new_labels = []
    new_sources = []

    model = AutoModelForMaskedLM.from_pretrained(config["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    for i in tqdm(range(len(dataframe))):
        data = dataframe.iloc[i]
        if data["label"] in label_list:
            index_id = data["id"]
            sentence = data["sentence"]
            subject_entity = data["subject_entity"]
            object_entity = data["object_entity"]
            label = data["label"]
            source = data["source"]

            text_data = []

            sentence = sentence.split()

            mask_text = sentence[:i] + ["[MASK]"] + sentence[i:]
            mask_text = " ".join(mask_text)
            text_data.append(mask_text)

            tokenized_text = tokenizer(
                text_data,
                return_tensors=config["tokenizers"]["return_tensors"],
                padding=config["tokenizers"]["padding"],
                truncation=config["tokenizers"]["truncation"],
                max_length=config["tokenizers"]["maxlength"],
                add_special_tokens=config["tokenizers"]["add_special_tokens"],
            )

            mask_idx_list = []
            tokens_list = []
            output = model(**tokenized_text)
            logits = output.logits

            for tokens in tokenized_text["input_ids"].tolist():
                mask_id = tokenizer.mask_token_id
                mask_idx = tokens.index(mask_id)
                mask_idx_list.append(mask_idx)
                tokens_list.append(tokens)
            count = 0
            for idx, mask_idx in enumerate(mask_idx_list):
                logits_pred = torch.argmax(F.softmax(logits[idx]), dim=1)
                mask_probablities = list(map(float, F.softmax(logits[idx])[mask_idx]))
                for i in range(len(mask_probablities)):
                    mask_probablities[i] = (mask_probablities[i], i)
                mask_probablities.sort(reverse=True)
                for i in range(3):
                    tokens_list[idx][mask_idx] = int(mask_probablities[i][1])
                    new_sentence = tokenizer.decode(torch.LongTensor(tokens_list[idx]))
                    new_sentence = new_sentence.strip("[SEP]")
                    new_sentence = new_sentence.strip("[CLS]")
                    new_sentence = new_sentence.strip("[PAD]")
                    new_sentence = new_sentence.strip()

                    subj = eval(subject_entity)["word"]
                    obj = eval(object_entity)["word"]
                    sub_type = eval(object_entity)["type"]
                    obj_type = eval(object_entity)["type"]

                    subj_indices = find_word_indices(new_sentence, subj)
                    obj_indices = find_word_indices(new_sentence, obj)

                    for sub_start_idx, sub_end_idx in subj_indices:
                        for obj_start_idx, obj_end_idx in obj_indices:
                            new_subj_entity = str(
                                {
                                    "word": subj,
                                    "start_idx": sub_start_idx,
                                    "end_idx": sub_end_idx,
                                    "type": sub_type,
                                }
                            )
                            new_obj_entity = str(
                                {
                                    "word": obj,
                                    "start_idx": obj_start_idx,
                                    "end_idx": obj_end_idx,
                                    "type": obj_type,
                                }
                            )
                            new_index_ids.append(index_id)
                            new_sentences.append(new_sentence)
                            new_subject_entities.append(new_subj_entity)
                            new_object_entities.append(new_obj_entity)
                            new_labels.append(label)
                            new_sources.append(source)

    new_data = pd.DataFrame(
        {
            "id": new_index_ids,
            "sentence": new_sentences,
            "subject_entity": new_subject_entities,
            "object_entity": new_object_entities,
            "label": new_labels,
            "souce": new_sources,
        }
    )

    new_data.to_csv("./generated_data.csv", index=True)
