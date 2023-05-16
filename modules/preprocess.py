import pandas as pd


def preprocessing_dataset(dataset : pd.DataFrame) -> pd.DataFrame:
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""

    subject_entity = []
    subject_s_idx = []
    subject_e_idx = []
    subject_entity_type = []

    object_entity = []
    object_s_idx = []
    object_e_idx = []
    object_entity_type = []

    for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
        sub_dict, obj_dict = eval(i), eval(j)

        subject = sub_dict["word"]
        sub_s_idx = sub_dict["start_idx"]
        sub_e_idx = sub_dict["end_idx"]
        sub_type = sub_dict["type"]

        object = obj_dict["word"]
        obj_s_idx = obj_dict["start_idx"]
        obj_e_idx = obj_dict["end_idx"]
        obj_type = obj_dict["type"]

        subject_entity.append(subject)
        subject_s_idx.append(sub_s_idx)
        subject_e_idx.append(sub_e_idx)
        subject_entity_type.append(sub_type)

        object_entity.append(object)
        object_s_idx.append(obj_s_idx)
        object_e_idx.append(obj_e_idx)
        object_entity_type.append(obj_type)
        
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'subject_start_idx':subject_s_idx,'subject_end_idx':subject_e_idx,'subject_entity_type':subject_entity_type,'object_entity':object_entity,'object_start_idx': object_s_idx,'object_end_idx':object_e_idx,'object_entity_type':object_entity_type,'label':dataset['label'],})

    return out_dataset
