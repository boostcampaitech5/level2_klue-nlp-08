import argparse
import os
import re
from datetime import datetime

import omegaconf
import pytz
import sklearn.metrics
from torch.optim.lr_scheduler import StepLR


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

def lr_scheduler(lr_scheduler_type, optimizer):
   if lr_scheduler_type == "stepLR":
      return StepLR(optimizer, step_size=1)
   # TODO 이외 lr scheduler 추가
   else:
      raise ValueError("정의되지 않은 lr scheduler type입니다.")

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
