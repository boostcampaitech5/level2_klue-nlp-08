import argparse

from ensembles import Ensemble
from utils import config_parser

if __name__ == "__main__":
    """
    아래는 mode에 대한 설명입니다. 읽고 parser로 인자를 넘겨주어 앙상블을 진행해주세요
    mode :
        esnb.yaml 파일을 수정해주세요.
    """
    config = config_parser()
    ensemble_model = Ensemble(config)

    if config["mode"] == "hard_voting":
        ensemble_model.ensemble_with_hard_voting()
        ensemble_model.save()
    elif config["mode"] == "possibility":
        ensemble_model.ensemble_with_possiblities_avg()
        ensemble_model.save()
    elif config["mode"] == "weighted_score":
        ensemble_model.ensemble_with_public_f1_score_weighted()
        ensemble_model.save()
    else:
        print("mode를 입력해주세요!")
