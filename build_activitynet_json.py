import json
from pathlib import Path
import numpy as np
import argparse

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--supervised", action="store_true")
args = parser.parse_args()

data_root = Path('data')
save_root = Path('sa_dataset')
if not save_root.exists():
    save_root.mkdir()
annots_all = {"database": {}}
for annot_file in data_root.glob('*.json'):
    with open(annot_file) as f:
        annot_data = json.load(f)
    video_name = Path(annot_data['metadata']['name']).stem
    video_url = annot_data['metadata']['url']
    if annot_data['metadata'].get('duration') is None:
        if not args.supervised:
            video_duration, video_resolution = '',''
            video_subset = 'validation'
            video_annotations = []
        else:
            continue
    else:
        video_duration = annot_data['metadata']['duration'] / 1000000
        video_width = annot_data['metadata']['width']
        video_height = annot_data['metadata']['height']
        video_resolution = '{}x{}'.format(video_width, video_height)
        video_subset = np.random.choice(['training', 'validation'], p=[0.9, 0.1])
        video_annotations = []
        for annot_instance in annot_data['instances']:
            if annot_instance['meta']['type'] == 'event':
                event_start = annot_instance['meta']['start'] / 1000000
                event_end = annot_instance['meta']['end'] / 1000000
                video_event = {
                    "label": annot_instance['meta']['className'],
                    "segment": [event_start, event_end]
                }
                video_annotations.append(video_event)
        if len(video_annotations) == 0:
            video_subset = 'validation'
    if (len(video_annotations) == 0) ^ args.supervised:
        annots_all['database'][video_name] = {
            "duration_second": video_duration,
            "subset": video_subset,
            "resolution": video_resolution,
            "url": video_url,
            "annotations": video_annotations
        }


with open(save_root / 'sa_dataset_activitynet.json', 'w') as f:
    json.dump(annots_all, f)

with open(data_root / "classes" / "classes.json") as f:
    class_data = json.load(f)

class_names = []
for class_entry in class_data:
    class_names.append(class_entry["name"])

with open("action_name.csv", "w") as f:
    f.write("action\n")
    for class_name in class_names:
        f.write(class_name + '\n')