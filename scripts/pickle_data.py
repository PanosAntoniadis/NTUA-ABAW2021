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


def frames_to_label_cat_all(expression, frames):
    detected_frames_ids = [int(frame.split('/')[-1].split('.')[0]) - 1 for frame in frames]
    all_frames_ids = [i for i in range(len(expression)) if expression[i]!=-1]
    indexes = [True if i in all_frames_ids else False for i in range(len(expression))]
    expression = expression[indexes]
    prefix = '/'.join(frames[0].split('/')[:-1])
    return_frames = []
    for id in all_frames_ids:
        if id in detected_frames_ids:
            return_frames.append(prefix+'/{0:05d}.jpg'.format(id+1))
        else:
            return_frames.append("not_detected")
    return expression, return_frames, all_frames_ids

def frames_to_label_cont_all(va, frames):
    detected_frames_ids = [int(frame.split('/')[-1].split('.')[0]) - 1 for frame in frames]
    all_frames_ids = [i for i in range(len(va)) if va[i][0]!=-5 and va[i][1]!=-5]
    indexes = [True if i in all_frames_ids else False for i in range(len(va))]
    va = va[indexes]
    prefix = '/'.join(frames[0].split('/')[:-1])
    return_frames = []
    for id in all_frames_ids:
        if id in detected_frames_ids:
            return_frames.append(prefix+'/{0:05d}.jpg'.format(id+1))
        else:
            return_frames.append("not_detected")
    return va[:, 0], va[:, 1], return_frames, all_frames_ids
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Process annotations of Aff-Wild2 database (train/val set)")
    parser.add_argument('--annotations_dir', type=str)
    parser.add_argument('--videos_dir', type=str)

    args = parser.parse_args()
    annotations_dir = args.annotations_dir
    videos_dir = args.videos_dir

    train_cat = os.path.join(annotations_dir, 'EXPR_Set/Train_Set')
    val_cat = os.path.join(annotations_dir, 'EXPR_Set/Validation_Set')
    train_cont = os.path.join(annotations_dir, 'VA_Set/Train_Set')
    val_cont = os.path.join(annotations_dir, 'VA_Set/Validation_Set')

    train_cat_names = []
    val_cat_names = []
    train_cont_names = []
    val_cont_names = []

    for filename in os.listdir(train_cat):
        train_cat_names.append(filename[:-4])
    for filename in os.listdir(val_cat):
        val_cat_names.append(filename[:-4])
    for filename in os.listdir(train_cont):
        train_cont_names.append(filename[:-4])
    for filename in os.listdir(val_cont):
        val_cont_names.append(filename[:-4])

    train_cat_values = []
    val_cat_values = []
    for filename in train_cat_names:
        with open(os.path.join(train_cat, filename) + '.txt') as f:
            next(f)
            train_cat_values.append(np.array([int(line.strip('\n')) for line in f]))
    for filename in val_cat_names:
        with open(os.path.join(val_cat, filename) + '.txt') as f:
            next(f)
            val_cat_values.append(np.array([int(line.strip('\n')) for line in f]))

    train_cont_values = []
    val_cont_values = []
    for filename in train_cont_names:
        with open(os.path.join(train_cont, filename) + '.txt') as f:
            next(f)
            x = []
            for line in f:
                x.append(list(map(float, line.strip('\n').split(','))))
            train_cont_values.append(np.array(x))
    for filename in val_cont_names:
        with open(os.path.join(val_cont, filename) + '.txt') as f:
            next(f)
            x = []
            for line in f:
                x.append(list(map(float, line.strip('\n').split(','))))
            val_cont_values.append(np.array(x))

    data = {}
    data_train_cat = []
    data_val_cat = []

    for i, filename in enumerate(train_cat_names):
        frames_paths = sorted(glob.glob(os.path.join(videos_dir, filename, '*.jpg')))
        video_path = os.path.join(videos_dir.replace("cropped_aligned","videos/both_batches"),filename + ".mp4").replace("_right","").replace("_left","")

        if not os.path.exists(video_path):
            video_path = video_path.replace(".mp4",".avi")
            if not os.path.exists(video_path):
                print(video_path)
        v = cv2.VideoCapture(video_path)

        expression_array, frames_paths, frames_ids = frames_to_label_cat_all(train_cat_values[i], frames_paths)
        frames = []
        if len(frames_paths) == 0:
            continue
        for j in range(len(frames_ids)):
            sample = Frame_annotation_cat(frame_path = frames_paths[j], expression = expression_array[j], frame_id = frames_ids[j])
            frames.append(sample)
        data_train_cat.append(Video_annotation(video_path = os.path.join(videos_dir, filename), frames_list = frames, fps=v.get(cv2.CAP_PROP_FPS)))
        v.release()
    
    for i, filename in enumerate(val_cat_names):
        frames_paths = sorted(glob.glob(os.path.join(videos_dir, filename, '*.jpg')))
        video_path = os.path.join(videos_dir.replace("cropped_aligned","videos/both_batches"),filename + ".mp4").replace("_right","").replace("_left","")

        if not os.path.exists(video_path):
            video_path = video_path.replace(".mp4",".avi")
            if not os.path.exists(video_path):
                print(video_path)
                raise
        v = cv2.VideoCapture(video_path)

        expression_array, frames_paths, frames_ids = frames_to_label_cat_all(val_cat_values[i], frames_paths)
        frames = []
        if len(frames_paths) == 0:
            continue
        for j in range(len(frames_ids)):
            sample = Frame_annotation_cat(frame_path = frames_paths[j], expression = expression_array[j], frame_id = frames_ids[j])
            frames.append(sample)
        data_val_cat.append(Video_annotation(video_path = os.path.join(videos_dir, filename), frames_list = frames, fps=v.get(cv2.CAP_PROP_FPS)))
        v.release()


    data_train_cont = []
    data_val_cont = []

    for i, filename in enumerate(train_cont_names):
        frames_paths = sorted(glob.glob(os.path.join(videos_dir, filename, '*.jpg')))
        video_path = os.path.join(videos_dir.replace("cropped_aligned","videos/both_batches"),filename + ".mp4").replace("_right","").replace("_left","")

        if not os.path.exists(video_path):
            video_path = video_path.replace(".mp4",".avi")
            if not os.path.exists(video_path):
                print(video_path)
                raise
        v = cv2.VideoCapture(video_path)

        valence_array, arousal_array, frames_paths, frames_ids = frames_to_label_cont_all(train_cont_values[i], frames_paths)
        frames = []
        if len(frames_paths) == 0:
            continue
        for j in range(len(frames_ids)):
            sample = Frame_annotation_cont(frame_path = frames_paths[j], valence=valence_array[j], arousal=arousal_array[j], frame_id = frames_ids[j])
            frames.append(sample)
        data_train_cont.append(Video_annotation(video_path = os.path.join(videos_dir, filename), frames_list = frames, fps=v.get(cv2.CAP_PROP_FPS)))
        v.release()
    
    for i, filename in enumerate(val_cont_names):
        frames_paths = sorted(glob.glob(os.path.join(videos_dir, filename, '*.jpg')))
        video_path = os.path.join(videos_dir.replace("cropped_aligned","videos/both_batches"),filename + ".mp4").replace("_right","").replace("_left","")

        if not os.path.exists(video_path):
            video_path = video_path.replace(".mp4",".avi")
            if not os.path.exists(video_path):
                print(video_path)
                raise
        v = cv2.VideoCapture(video_path)

        valence_array, arousal_array, frames_paths, frames_ids = frames_to_label_cont_all(val_cont_values[i], frames_paths)
        frames = []
        if len(frames_paths) == 0:
            continue
        for j in range(len(frames_ids)):
            sample = Frame_annotation_cont(frame_path = frames_paths[j], valence=valence_array[j], arousal=arousal_array[j], frame_id = frames_ids[j])
            frames.append(sample)
        data_val_cont.append(Video_annotation(video_path = os.path.join(videos_dir, filename), frames_list = frames, fps=v.get(cv2.CAP_PROP_FPS)))
        v.release()

    data = {'train_cat': data_train_cat, 'val_cat': data_val_cat, 'train_cont': data_train_cont, 'val_cont': data_val_cont}

    with open('data.pkl', "wb") as w:
            pickle.dump(data, w)
