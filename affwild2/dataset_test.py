import os
import cv2
import math
import torch
import pickle
import numpy as np
import random
import torchaudio
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset
from affwild2.dataset import Video_annotation


class Video_dataset_cat_test(Dataset):
    def __init__(self, data_pkl, transform=None, duration=90):
        self.transform = transform
        self.duration = duration

        data_pickle = pickle.load(open(data_pkl, 'rb'))
        data_old = data_pickle['test_cat']    
        data_new = []
        for d in data_old:
            frames = d.frames_list
            for i in range(0, len(frames), self.duration):
                end = min(i+self.duration, len(frames))
                data_new.append(Video_annotation(video_path=d.video_path, frames_list=frames[i:end], fps=d.video_fps))

        self.data = data_new

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        samples = self.data[idx].frames_list
        video_name = self.data[idx].video_path.split('/')[-1]
        frames = []
        valid = []
        frame_ids = []

        if len(samples) < self.duration:
            for i in range(self.duration):
                if samples[i%len(samples)].frame_path == "not_detected":
                    image = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                else:
                    image = Image.open(samples[i%len(samples)].frame_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)

                frame_ids.append(samples[i%len(samples)].frame_id)
                frames.append(image)
                valid.append(i<len(samples))
        else:
            for i in range(self.duration):
                if samples[i].frame_path == "not_detected":
                    image = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                else:
                    image = Image.open(samples[i].frame_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)

                frame_ids.append(samples[i].frame_id)
                frames.append(image)
                valid.append(True)

        frames = torch.stack(frames)
        valid = torch.BoolTensor(valid)

        return frames, valid, video_name



class Video_dataset_cont_test(Dataset):
    def __init__(self, data_pkl, transform=None, duration=16):
        self.transform = transform
        self.duration = duration
        data_pickle = pickle.load(open(data_pkl, 'rb'))
        data_old = data_pickle['test_cont']
        
        data_new = []
        for d in data_old:
            frames = d.frames_list
            for i in range(0, len(frames), self.duration):
                end = min(i+self.duration, len(frames))
                data_new.append(Video_annotation(video_path=d.video_path, frames_list=frames[i:end], fps=d.video_fps))

        self.data = data_new

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        samples = self.data[idx].frames_list
        video_name = self.data[idx].video_path.split('/')[-1]

        frames = []
        valid = []

        if len(samples) < self.duration:
            for i in range(self.duration):
                if samples[i%len(samples)].frame_path == "not_detected":
                    image = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                else:
                    image = Image.open(samples[i%len(samples)].frame_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
                valid.append(i<len(samples))
        else:
            for i in range(self.duration):
                if samples[i].frame_path == "not_detected":
                    image = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                else:
                    image = Image.open(samples[i].frame_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
                valid.append(True)

        frames = torch.stack(frames)
        valid = torch.BoolTensor(valid)
        return frames, valid, video_name
