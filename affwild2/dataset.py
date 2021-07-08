import os
import cv2
import math
import torch
import pickle
import torchaudio
import numpy as np
import random
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset
from affwild2.ops.utils import get_histogram


class Frame_annotation_cat(object):
    def __init__(self, frame_path, frame_id, expression):
        super(Frame_annotation_cat, self).__init__()
        self.frame_path = frame_path
        self.frame_id = frame_id
        self.expression = expression

class Frame_annotation_cont(object):
    def __init__(self, frame_path, frame_id, valence, arousal):
        super(Frame_annotation_cont, self).__init__()
        self.frame_path = frame_path
        self.frame_id = frame_id
        self.valence = valence
        self.arousal = arousal

class Video_annotation(object):
    def __init__(self, video_path, frames_list, fps):
        super(Video_annotation, self).__init__()
        self.video_path = video_path
        self.video_fps = fps
        self.frames_list = frames_list

class Video_dataset_cat(Dataset):
    def __init__(self, data_pkl, train=True, transform=None, duration=16, audio=False, audio_transform=None):
        self.train = train
        self.transform = transform
        self.duration = duration
        self.audio = audio
        self.audio_transform = audio_transform

        data_pickle = pickle.load(open(data_pkl, 'rb'))
        if train:
            self.data = data_pickle['train_cat']
        else:
            self.data = data_pickle['val_cat']

        data_new = []
        for d in self.data:
            frames = d.frames_list
            for i in range(0, len(frames), self.duration):
                end = min(i+self.duration, len(frames))
                data_new.append(Video_annotation(video_path=d.video_path, frames_list=frames[i:end], fps=d.video_fps))
        self.data = data_new
        
        # audio params
        if self.audio:
            self.window_size = 20e-3
            self.window_stride = 10e-3
            self.sample_rate = 44100
            num_fft = 2 ** math.ceil(math.log2(self.window_size * self.sample_rate))
            window_fn = torch.hann_window

            self.sample_len_secs = 10
            self.sample_len_frames = self.sample_len_secs * self.sample_rate
            self.audio_shift_sec = 5
            self.audio_shift_samples = self.audio_shift_sec * self.sample_rate
            # transforms

            self.audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_mels=64,
                                                                        n_fft=num_fft,
                                                                        win_length=int(self.window_size * self.sample_rate),
                                                                        hop_length=int(self.window_stride
                                                                                       * self.sample_rate),
                                                                        window_fn=window_fn)
            self.audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])#, torchaudio.transforms.TimeMasking(100), torchaudio.transforms.FrequencyMasking(10)])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        samples = self.data[idx].frames_list
        frames = []
        expressions = []
        valid = []
        frame_ids = []

        if len(samples) < self.duration:
            for i in range(self.duration):
                expression = samples[i%len(samples)].expression
                if samples[i%len(samples)].frame_path == "not_detected":
                    image = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                else:
                    image = Image.open(samples[i%len(samples)].frame_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                
                frames.append(image)
                frame_ids.append(samples[i%len(samples)].frame_id)
                expressions.append(expression)
                valid.append(i < len(samples))
                    
        else:
            start = random.randint(0, len(samples)-self.duration)
            end = start+self.duration#*stride
            for i in range(start, end, 1):
                expression = samples[i].expression
                if samples[i].frame_path == "not_detected":
                    image = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                else:
                    image = Image.open(samples[i].frame_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)

                frame_ids.append(samples[i].frame_id)
                frames.append(image)
                expressions.append(expression)
                valid.append(True)

    
        frames = torch.stack(frames)
        expressions = torch.LongTensor(expressions)
        valid = torch.BoolTensor(valid)

        # ---------------------- handle audio -------------------------- #
        if self.audio:
            fps = self.data[idx].video_fps

            audio_file = "/gpu-data3/filby/aff-wild2/wav_tnt/" + self.data[idx].video_path.split("/")[-1].split(".")[0] + ".wav"
            audio_file = audio_file.replace("_right","").replace("_left","")

            audio_frames = []

            for i in frame_ids:
                frame_offset = i/fps*44100 - 5 * 44100
                if frame_offset < 0:
                    frame_offset = 0

                num_frames = 10*44100

                audioData, sampleRate = torchaudio.load(audio_file, offset=int(frame_offset), num_frames=int(num_frames))

                assert sampleRate == 44100

                audio = self.audio_spec_transform(self.audio_transform(audioData))

                if audio.size(2) != 1001:
                    audio = F.pad(audio,[0, 1001-audio.size(2)])

                audio_frames.append(audio)

            return frames, expressions, valid, torch.stack(audio_frames)
        else:
            return frames, expressions, valid


class Video_dataset_cont(Dataset):
    def __init__(self, data_pkl, train=True, transform=None, duration=16):
        self.train = train
        self.transform = transform
        self.duration = duration
        data_pickle = pickle.load(open(data_pkl, 'rb'))
        if train:
            self.data = data_pickle['train_cont']
        else:
            self.data = data_pickle['val_cont']

        data_new = []
        for d in self.data:
            frames = d.frames_list
            for i in range(0, len(frames), self.duration):
                end = min(i+self.duration, len(frames))
                data_new.append(Video_annotation(video_path=d.video_path, frames_list=frames[i:end], fps=d.video_fps))
        self.data = data_new

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        samples = self.data[idx].frames_list
        frames = []
        valence = []
        arousal = []
        valid = []
        
        if len(samples) < self.duration:
            for i in range(self.duration):
                val = samples[i%len(samples)].valence
                ar = samples[i%len(samples)].arousal
                if samples[i%len(samples)].frame_path == "not_detected":
                    image = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                else:
                    image = Image.open(samples[i%len(samples)].frame_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
                valence.append(val)
                arousal.append(ar)
                valid.append(i < len(samples))
        else:
            start = random.randint(0, len(samples)-self.duration)
            end = start+self.duration
            for i in range(start, end):
                val = samples[i].valence
                ar = samples[i].arousal
                if samples[i].frame_path == "not_detected":
                    image = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                else:
                    image = Image.open(samples[i].frame_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
                valence.append(val)
                arousal.append(ar)
                valid.append(i < len(samples))

        frames = torch.stack(frames)
        valence = torch.FloatTensor(valence)
        arousal = torch.FloatTensor(arousal)
        cont = torch.stack([valence, arousal], dim=1)
        valid = torch.BoolTensor(valid)
        return frames, cont, valid

