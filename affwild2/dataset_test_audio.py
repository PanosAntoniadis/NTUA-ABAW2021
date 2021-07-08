import os
import cv2
import torch
import pickle
import numpy as np
import random
import math, torchaudio
from PIL import Image
from torch.utils.data import Dataset
from affwild2.dataset import Video_annotation
import torchvision.transforms as transforms
from .audio_utils import get_melspectrogram_db
from .tnt_transforms import *
import torch.nn.functional as F


class Video_dataset_cat_test(Dataset):
    def __init__(self, data_pkl, transform=None, duration=90):
        self.transform = transform
        self.duration = duration

        data_pickle = pickle.load(open(data_pkl, 'rb'))
        data_old = data_pickle['val_cat']
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
                # if samples[i%len(samples)].frame_path == "not_detected":
                #     image = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                # else:
                #     image = Image.open(samples[i%len(samples)].frame_path).convert("RGB")
                # if self.transform:
                #     image = self.transform(image)

                frame_ids.append(samples[i%len(samples)].frame_id)
                # frames.append(image)
                valid.append(i<len(samples))
        else:
            for i in range(self.duration):
                # if samples[i].frame_path == "not_detected":
                #     image = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                # else:
                #     image = Image.open(samples[i].frame_path).convert("RGB")
                # if self.transform:
                #     image = self.transform(image)

                frame_ids.append(samples[i].frame_id)
                # frames.append(image)
                valid.append(True)

        fps = self.data[idx].video_fps

        mel_file = "/gpu-data3/filby/aff-wild2/wav_melspec/" + self.data[idx].video_path.split("/")[-1].split(".")[
            0] + ".npy"
        mel_file = mel_file.replace("_right", "").replace("_left", "")
        mel = np.load(mel_file)

        audio_frames = []

        window_stride = 512 / 44100  # seconds

        # mel spec
        for i in frame_ids:
            seconds_video_offset = i / fps - 5
            if seconds_video_offset < 0:
                seconds_video_offset = 0

            frame_offset = int(seconds_video_offset / window_stride)

            l = int(10 / window_stride)
            audio_features = mel[:, :, frame_offset:frame_offset + l]

            # audio_features = self.scaler.transform(audio_features.T).T

            audio_features = torch.Tensor(audio_features)  # .unsqueeze(0)
            # print(audio_features.size())
            if audio_features.size(2) != l:
                audio_features = F.pad(audio_features, [0, l - audio_features.size(2)])

            # audio_features = self.audio_transform(audio_features)

            audio_frames.append(audio_features)

        audio = torch.stack(audio_frames)

        # frames = torch.stack(frames)
        valid = torch.BoolTensor(valid)

        return audio, valid, video_name



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
