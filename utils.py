import argparse

import omegaconf
import sklearn.metrics
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, StepLR

# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


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
      return StepLR(optimizer, step_size=1, verbose=True)
   elif lr_scheduler_type == "exponentialLR":
      return ExponentialLR(optimizer, gamma = 0.9, verbose=True)
   elif lr_scheduler_type == "lambdaLR":
      return LambdaLR(optimizer, lr_lambda=lambda epoch: 0.8 ** epoch, verbose=True)
   # TODO 이외 lr scheduler 추가
   else:
      raise ValueError(f"{lr_scheduler_type} : 정의되지 않은 lr scheduler type입니다.")
