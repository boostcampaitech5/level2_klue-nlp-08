import os
import sys

print(os.path.dirname(__file__))
print(os.path.abspath(os.path.dirname(__file__)))
print(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ensemble_model import Ensemble
from modules.utils import config_parser

if __name__ == "__main__":
    config = config_parser()
    ensemble_model = Ensemble(config)

    if config["mode"] == "hard_voting":
        ensemble_model.ensemble_with_hard_voting()
        ensemble_model.save()
    elif config["mode"] == "soft_voting":
        ensemble_model.ensemble_with_possiblities_avg()
        ensemble_model.save()
    elif config["mode"] == "weighted_score":
        ensemble_model.ensemble_with_public_f1_score_weighted()
        ensemble_model.save()
    else:
        print("올바른 mode를 입력해주세요! hard")
