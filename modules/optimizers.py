import torch.optim as optim
from adabelief_pytorch import AdaBelief

def get_optimizer(optimizer_type):
    if optimizer_type == "AdamW":
        return optim.AdamW
    elif optimizer_type == "Adam":
        return optim.Adam
    elif optimizer_type == "SGD":
        return optim.SGD
    elif optimizer_type == "Adabelief":
        return AdaBelief
    else:
      raise ValueError("정의되지 않은 optimizer type입니다.")
