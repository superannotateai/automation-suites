# Copyright (c) OpenMMLab. All rights reserved.
"""This file processes the annotation files and generates proper annotation
files for localizers."""
import json
from pathlib import Path
import numpy as np

base_dir = Path('./sa_dataset')

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

ann_file = base_dir / "sa_dataset_activitynet.json"

anno_database = load_json(ann_file)

video_dict_train = {}
video_dict_val = {}
video_dict_test = {}

for video_name, video_data in anno_database['database'].items():
    video_subset = video_data["subset"]
    video_data.pop("subset")
    if video_subset == 'training':
        video_dict_train[video_name] = video_data
    elif video_subset == 'testing':
        video_dict_test[video_name] = video_data
    elif video_subset == 'validation':
        video_dict_val[video_name] = video_data

print(f'full subset video numbers: {len(anno_database["database"].keys())}')

with open(base_dir / 'sa_dataset_train.json', 'w') as result_file:
    json.dump(video_dict_train, result_file)

with open(base_dir / 'sa_dataset_val.json', 'w') as result_file:
    json.dump(video_dict_val, result_file)

with open(base_dir / 'sa_dataset_test.json', 'w') as result_file:
    json.dump(video_dict_test, result_file)
