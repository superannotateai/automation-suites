# Copyright (c) OpenMMLab. All rights reserved.
# This scripts is copied from
# https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/download.py  # noqa: E501
# The code is licensed under the MIT licence.
import argparse
import os
import ssl
import subprocess

import mmcv
from joblib import Parallel, delayed

ssl._create_default_https_context = ssl._create_unverified_context
data_file = './sa_dataset'
output_dir = f'{data_file}/videos'

def get_length(input_video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

def download_clip(video_url,
                  output_filename,
                  num_attempts=5):
    """Download a video from youtube if exists and is not blocked.
    arguments:
    ---------
    video_url: str
        Unique YouTube video identifier (11 characters)
    output_filename: str
        File path where the video will be stored.
    """
    # Defensive argument checking.
    assert isinstance(video_url, str), 'video_identifier must be string'
    assert isinstance(output_filename, str), 'output_filename must be string'

    status = False

    if not os.path.exists(output_filename):
        command = [
            'youtube-dl', '--quiet', '--no-warnings', '--no-check-certificate',
            '-f', 'mp4', '-o',
            '"%s"' % output_filename,
            '"%s"' % (video_url)
        ]
        command = ' '.join(command)
        print(command)
        attempts = 0
        while True:
            try:
                subprocess.check_output(
                    command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                attempts += 1
                if attempts == num_attempts:
                    return status, 'Fail'
            else:
                break
    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'


def download_clip_wrapper(id_url, output_dir):
    """Wrapper for parallel processing purposes."""
    # we do this to align with names in annotations
    output_filename = os.path.join(output_dir, id_url[0] + '.mp4')
    if os.path.exists(output_filename):
        status = tuple([id_url[0], True, 'Exists'])
        return status

    downloaded, log = download_clip(id_url[1], output_filename)
    status = tuple([id_url[0], downloaded, log])
    duration = get_length(output_filename)

    return (status, id_url[0], duration)


def parse_activitynet_annotations(anno_file):
    """Returns a list of YoutubeID.
    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'video,numFrame,seconds,fps,rfps,subset,featureFrame'
    returns:
    -------
    youtube_ids: list
        List of all YoutubeIDs in ActivityNet.

    """

    data = mmcv.load(anno_file)
    idx_urls = []
    for id, data_point in data['database'].items():
        idx_urls.append((id, data_point['url']))

    return idx_urls, data


def main(anno_file, output_dir, num_jobs=24):
    # Reading and parsing ActivityNet.
    youtube_ids, data = parse_activitynet_annotations(anno_file)

    # Creates folders where videos will be saved later.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Download all clips.
    if num_jobs == 1:
        status_list = []
        for index in youtube_ids:
            status_list.append(download_clip_wrapper(index, output_dir))
    else:
        status_list = Parallel(n_jobs=num_jobs)(
            delayed(download_clip_wrapper)(index, output_dir)
            for index in youtube_ids)
    
    only_statuses = []
    for status, index, duration in status_list:
        if data['database'][index]['duration_second'] == "":
            data['database'][index]['duration_second'] = duration
        only_statuses.append(status)
    # Save download report.
    mmcv.dump(only_statuses, 'download_report.json')
    mmcv.dump(data, anno_file)


if __name__ == '__main__':
    video_list = f'{data_file}/sa_dataset_activitynet.json'
    anno_file = video_list
    main(video_list, output_dir, 24)