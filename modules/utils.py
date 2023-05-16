import argparse
import os
import pickle
import re
from datetime import datetime
from typing import Dict, List

import omegaconf
import pytz
import sklearn.metrics
import torch
from transformers import BatchEncoding


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def config_parser():
   parser = argparse.ArgumentParser()
   parser.add_argument('--config', type = str, default = 'config/default.yaml')
   args = parser.parse_args()

   config = omegaconf.OmegaConf.load(args.config)
   return config

def show_confusion_matrix(preds, labels, epoch, save_path):
   now = datetime.now(pytz.timezone("Asia/Seoul"))

   matrix = sklearn.metrics.confusion_matrix(y_pred=preds, y_true=labels)
   os.makedirs("confusion_matrix", exist_ok=True)
   os.makedirs(save_path, exist_ok=True)
   with open(f"{save_path}/epoch:{epoch}&{now.strftime('%Y-%m-%d %H.%M.%S')}.txt", "w") as f:
      content = str(matrix)
      content = re.sub(r'\s+', ',', content)
      content = re.sub(r'\d+', lambda m: f"{m.group(0):>4}", content)
      content = re.sub(r'\],\[,', '\n', content)
      content = content[2:-2]
      f.write(content)

def label_to_num(label : str) -> int:
      with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "pickle", 'dict_label_to_num.pkl'), 'rb') as f:
         dict_label_to_num = pickle.load(f)
      return dict_label_to_num[label]

def num_to_label(label : List[int]) -> List[str]:
   origin_label = []
   with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickle", 'dict_num_to_label.pkl'), 'rb') as f:
      dict_num_to_label = pickle.load(f)
   for v in label:
      origin_label.append(dict_num_to_label[v])
   
   return origin_label

def make_index_ids_roberta(tokenized_sentences: Dict) -> BatchEncoding:
      index_ids = []
      for i in range(len(tokenized_sentences.data['attention_mask'])):
         index_number = 0
         temp_input_ids = tokenized_sentences.data['input_ids'][i]
         index_tensor = torch.zeros_like(temp_input_ids)

         for index, value in enumerate(temp_input_ids):
               if value == 1:
                  break
               if value >= 32000 and value <= 32005:
                  index_number = 1
               if value > 32005:
                  index_number = 0
               index_tensor[index] = index_number
               if value >= 32000:
                  index_tensor[index] = 0
         index_ids.append(list(index_tensor.to(dtype=torch.int64)))
      return torch.tensor(index_ids)

def get_special_token(dataset_type : str) -> List:
   if dataset_type == "default":
      return []
   elif dataset_type == "punct":
      return []
   elif dataset_type == "type_entity":
      return ["[ORG]", "[PER]", "[LOC]", "[POH]", "[DAT]", "[NOH]", "[/ORG]", "[/PER]", "[/LOC]", "[/POH]", "[/DAT]", "[/NOH]"]
   else:
      raise ValueError("정의되지 않은 dataset type입니다.")

