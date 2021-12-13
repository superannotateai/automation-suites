# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
import argparse

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--supervised", action="store_true")
args = parser.parse_args()

data_file = './sa_dataset'
rawframe_dir = f'{data_file}/rawframes'
action_name_list = 'action_name.csv'

train_rawframe_dir = rawframe_dir
val_rawframe_dir = rawframe_dir
test_rawframe_dir = rawframe_dir

json_file = f'{data_file}/sa_dataset_activitynet.json'


def generate_rawframes_filelist():
    with open(json_file) as f:
        load_dict = json.load(f)

    anet_labels = open(action_name_list).readlines()
    anet_labels = [x.strip() for x in anet_labels[1:]]

    train_dir_list = [
        osp.join(train_rawframe_dir, x) for x in os.listdir(train_rawframe_dir)
    ]
    val_dir_list = [
        osp.join(val_rawframe_dir, x) for x in os.listdir(val_rawframe_dir)
    ]
    test_dir_list = [
        osp.join(test_rawframe_dir, x) for x in os.listdir(test_rawframe_dir)
    ]

    def simple_label(anno):
        label = anno[0]['label']
        return anet_labels.index(label)

    def count_frames(dir_list, video):
        for dir_name in dir_list:
            if video in dir_name:
                return osp.basename(dir_name), len([im_name for im_name in os.listdir(dir_name) if im_name.startswith("img")])
        return None, None

    database = load_dict['database']
    training = {}
    validation = {}
    test = {}
    key_dict = {}

    for k in database:
        data = database[k]
        subset = data['subset']

        if subset in ['training', 'validation', 'test']:
            annotations = data['annotations']
            if subset != 'test':
                label = simple_label(annotations)
            else:
                label = -1
            if subset == 'training':
                dir_list = train_dir_list
                data_dict = training
            elif subset == 'test':
                dir_list = test_dir_list
                data_dict = test
            else:
                dir_list = val_dir_list
                data_dict = validation

        else:
            continue

        gt_dir_name, num_frames = count_frames(dir_list, k)
        if gt_dir_name is None:
            continue
        data_dict[gt_dir_name] = [num_frames, label]
        key_dict[gt_dir_name] = k
        load_dict['database'][k]['duration_frame'] = num_frames

    train_lines = [
        k + ' ' + str(training[k][0]) + ' ' + str(training[k][1])
        for k in training
    ]
    val_lines = [
        k + ' ' + str(validation[k][0]) + ' ' + str(validation[k][1])
        for k in validation
    ]
    test_lines = [
        k + ' ' + str(test[k][0]) + ' ' + str(test[k][1])
        for k in test
    ]

    with open(osp.join(data_file, 'anet_train_video.txt'), 'w') as fout:
        fout.write('\n'.join(train_lines))
    with open(osp.join(data_file, 'anet_val_video.txt'), 'w') as fout:
        fout.write('\n'.join(val_lines))
    with open(osp.join(data_file, 'anet_test_video.txt'), 'w') as fout:
        fout.write('\n'.join(test_lines))

    def clip_list(k, anno, video_anno):
        duration = anno['duration_second']
        num_frames = video_anno[0]
        fps = num_frames / duration
        segs = anno['annotations']
        lines = []
        for seg in segs:
            segment = seg['segment']
            label = seg['label']
            label = anet_labels.index(label)
            start, end = int(segment[0] * fps), int(segment[1] * fps)
            if end > num_frames - 1:
                end = num_frames - 1
            newline = f'{k} {start} {end - start + 1} {label}'
            lines.append(newline)
        return lines
    if args.supervised:
        train_clips, val_clips = [], []
        for k in training:
            train_clips.extend(clip_list(k, database[key_dict[k]], training[k]))
        for k in validation:
            val_clips.extend(clip_list(k, database[key_dict[k]], validation[k]))

        with open(osp.join(data_file, 'anet_train_clip.txt'), 'w') as fout:
            fout.write('\n'.join(train_clips))
        with open(osp.join(data_file, 'anet_val_clip.txt'), 'w') as fout:
            fout.write('\n'.join(val_clips))
    with open(json_file, 'w') as fout:
        json.dump(load_dict, fout)


if __name__ == '__main__':
    generate_rawframes_filelist()