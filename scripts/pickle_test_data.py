import os
import cv2
import glob
import math
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from affwild2.dataset import Frame_annotation_cat, Frame_annotation_cont, Video_annotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Process annotations of Aff-Wild2 database (test set)")
    parser.add_argument('--cat_videos', type=str)
    parser.add_argument('--cont_videos', type=str)
    parser.add_argument('--videos_dir', type=str)

    args = parser.parse_args()
    videos_dir = args.videos_dir
    cat_videos = args.cat_videos
    cont_videos = args.cont_videos

    cat_names = []
    cont_names = []
    
    with open(cat_videos) as f:
        for line in f:
            cat_names.append(line.strip('\n'))
    
    with open(cont_videos) as f:
        for line in f:
            cont_names.append(line.strip('\n'))

    data = {}
    data_cat = []
    data_cont = []

    for i, filename in enumerate(cat_names):
        frames_paths = sorted(glob.glob(os.path.join(videos_dir, filename, '*.jpg')))
        video_path = os.path.join(videos_dir.replace("cropped_aligned", "videos/both_batches"),filename + ".mp4").replace("_right","").replace("_left","")
        if not os.path.exists(video_path):
            video_path = video_path.replace(".mp4",".avi")
            if not os.path.exists(video_path):
                print(video_path)
        
        v = cv2.VideoCapture(video_path)
        length = int(v.get(cv2.CAP_PROP_FRAME_COUNT))

        detected_frames_ids = [int(frame.split('/')[-1].split('.')[0]) - 1 for frame in frames_paths]
        all_frames_ids = [i for i in range(length)]
        prefix = '/'.join(frames_paths[0].split('/')[:-1])

        all_frames_paths = []
        for id in all_frames_ids:
            if id in detected_frames_ids:
                all_frames_paths.append(prefix+'/{0:05d}.jpg'.format(id+1))
            else:
                all_frames_paths.append("not_detected")

        frames = []
        for j in range(length):
            sample = Frame_annotation_cat(frame_path = all_frames_paths[j], expression = None, frame_id = all_frames_ids[j])
            frames.append(sample)

        data_cat.append(Video_annotation(video_path = os.path.join(videos_dir, filename), frames_list = frames, fps=v.get(cv2.CAP_PROP_FPS)))
        v.release()

    for i, filename in enumerate(cont_names):
        frames_paths = sorted(glob.glob(os.path.join(videos_dir, filename, '*.jpg')))
        video_path = os.path.join(videos_dir.replace("cropped_aligned", "videos/both_batches"),filename + ".mp4").replace("_right","").replace("_left","")
        if not os.path.exists(video_path):
            video_path = video_path.replace(".mp4",".avi")
            if not os.path.exists(video_path):
                print(video_path)
        
        v = cv2.VideoCapture(video_path)
        length = int(v.get(cv2.CAP_PROP_FRAME_COUNT))

        detected_frames_ids = [int(frame.split('/')[-1].split('.')[0]) - 1 for frame in frames_paths]
        all_frames_ids = [i for i in range(length)]
        prefix = '/'.join(frames_paths[0].split('/')[:-1])

        all_frames_paths = []
        for id in all_frames_ids:
            if id in detected_frames_ids:
                all_frames_paths.append(prefix+'/{0:05d}.jpg'.format(id+1))
            else:
                all_frames_paths.append("not_detected")

        frames = []
        for j in range(length):
            sample = Frame_annotation_cont(frame_path = all_frames_paths[j], valence = None, arousal = None, frame_id = all_frames_ids[j])
            frames.append(sample)

        data_cont.append(Video_annotation(video_path = os.path.join(videos_dir, filename), frames_list = frames, fps=v.get(cv2.CAP_PROP_FPS)))
        v.release()
    

    data = {'test_cat': data_cat, 'test_cont': data_cont}

    with open('data_test.pkl', "wb") as w:
            pickle.dump(data, w)