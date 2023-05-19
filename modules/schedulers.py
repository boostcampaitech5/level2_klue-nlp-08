from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


def get_scheduler(scheduler_type):
   if scheduler_type == "stepLR":
      return StepLR
   # TODO 이외 lr scheduler 추가
   elif scheduler_type == "CosineLR":
      return CosineAnnealingLR
   else:
      raise ValueError("정의되지 않은 scheduler type입니다.")
