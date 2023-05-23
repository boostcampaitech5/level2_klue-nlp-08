import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils import list_argmax, softmax


class Ensemble:
    def __init__(self, config):
        self.data_list = []
        self.data_path_list = config["data_path_list"]
        self.model_f1_score = np.array(config["model_f1_score"])
        for data_path in self.data_path_list:
            self.data_list.append(pd.read_csv(data_path))
        self.df_number = len(self.data_list)
        self.ensemble = None
        self.save_path = config["save_path"]

    def ensemble_with_possiblities_avg(self):
        self.ensemble = pd.DataFrame()
        ensemble_pred = []
        ensemble_prob = []
        ensemble_id = []

        for i in range(self.df_number):
            self.ensemble[f"probs_candidate{i+1}"] = self.data_list[i]["probs"]

        for i in tqdm(range(len(self.ensemble)), desc="Making Soft Voting Ensemble"):
            probs = []
            new_prob = [0] * 30

            for j in range(self.df_number):
                probs.append(eval(self.ensemble.iloc[i][f"probs_candidate{j+1}"]))

            for prob in probs:
                for k in range(30):
                    new_prob[k] += prob[k]
            for k in range(30):
                new_prob[k] /= self.df_number

            pred, _ = list_argmax(new_prob)

            ensemble_pred.append(pred)
            ensemble_prob.append(new_prob)
            ensemble_id.append(i)

        ensemble_pred = self.num_to_label(ensemble_pred)
        self.ensemble = pd.DataFrame(
            {"id": ensemble_id, "pred_label": ensemble_pred, "probs": ensemble_prob}
        )
        return self.ensemble

    def ensemble_with_hard_voting(self):
        self.ensemble = pd.DataFrame()
        ensemble_pred = []
        ensemble_prob = []
        ensemble_id = []

        for i in range(self.df_number):
            self.ensemble[f"probs_candidate{i+1}"] = self.data_list[i]["probs"]

        for i in tqdm(range(len(self.ensemble)), desc="Making Hard Voting Ensemble"):
            probs = []
            new_prob = [0] * 30
            candidate = []
            pred_dict = {}
            pred_count = {}
            pred_locate = {}

            for j in range(self.df_number):
                probs.append(eval(self.ensemble.iloc[i][f"probs_candidate{j+1}"]))

            for j in range(self.df_number):
                pred, possibility = list_argmax(probs[j])
                if pred not in pred_locate:
                    pred_dict[pred] = possibility
                    pred_count[pred] = 1
                    pred_locate[pred] = [j]
                else:
                    pred_dict[pred] += possibility
                    pred_count[pred] += 1
                    pred_locate[pred].append(j)

            check_max = -1
            for key, value in pred_count.items():
                if value > check_max:
                    check_max = value
                    candidate = [key]
                elif value == check_max:
                    candidate.append(key)

            chosen = candidate[0]
            if len(candidate) > 1:
                choose_possibility = 0
                for c in candidate:
                    if pred_dict[c] >= choose_possibility:
                        chosen = c
                        choose_possibility = pred_dict[c]

            for key in pred_locate:
                if chosen == key:
                    for idx in pred_locate[key]:
                        for k in range(30):
                            new_prob[k] += probs[idx][k]
            for k in range(30):
                new_prob[k] /= len(pred_locate[chosen])

            pred, _ = list_argmax(new_prob)

            ensemble_pred.append(pred)
            ensemble_prob.append(new_prob)
            ensemble_id.append(i)

        ensemble_pred = self.num_to_label(ensemble_pred)
        self.ensemble = pd.DataFrame(
            {"id": ensemble_id, "pred_label": ensemble_pred, "probs": ensemble_prob}
        )
        return self.ensemble

    def ensemble_with_public_f1_score_weighted(self):
        self.ensemble = pd.DataFrame()
        ensemble_pred = []
        ensemble_prob = []
        ensemble_id = []
        exp_model_f1_score = list(softmax(self.model_f1_score))

        for i in range(self.df_number):
            self.ensemble[f"probs_candidate{i+1}"] = self.data_list[i]["probs"]

        for i in tqdm(
            range(len(self.ensemble)), desc="Making f1 score weighted Ensemble"
        ):
            probs = []
            new_prob = [0] * 30

            for j in range(self.df_number):
                probs.append(eval(self.ensemble.iloc[i][f"probs_candidate{j+1}"]))

            for j in range(self.df_number):
                for k in range(30):
                    new_prob[k] += probs[j][k] * exp_model_f1_score[j]

            pred, _ = list_argmax(new_prob)

            ensemble_pred.append(pred)
            ensemble_prob.append(new_prob)
            ensemble_id.append(i)

        ensemble_pred = self.num_to_label(ensemble_pred)
        self.ensemble = pd.DataFrame(
            {"id": ensemble_id, "pred_label": ensemble_pred, "probs": ensemble_prob}
        )
        return self.ensemble

    def num_to_label(self, label: List[int]) -> List[str]:
        """
        숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
        """
        origin_label = []
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "pickle",
                "dict_num_to_label.pkl",
            ),
            "rb",
        ) as f:
            dict_num_to_label = pickle.load(f)

        for v in label:
            origin_label.append(dict_num_to_label[v])
        return origin_label

    def save(self):
        self.ensemble.to_csv(self.save_path, index=False)
        print(f"Ensemble model Save Complete")
