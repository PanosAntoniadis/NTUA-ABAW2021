import os
import cv2
import json
import torch
import pickle
import random
import numpy as np
import math
import torchaudio
import torchvision.transforms as transforms


from PIL import Image
from torch.utils.data import Dataset
from affwild2.dataset import Video_annotation
import torch.nn.functional as F

def get_context(image, joints, format="PIL"):
    
    joints = np.reshape(joints, (25,3))
    joints[joints[:,2]<0.1] = np.nan
    joints[np.isnan(joints[:,2])] = np.nan

    joint_min_x = int(round(np.nanmin(joints[:,0])))
    joint_min_y = int(round(np.nanmin(joints[:,1])))

    joint_max_x = int(round(np.nanmax(joints[:,0])))
    joint_max_y = int(round(np.nanmax(joints[:,1])))

    expand_x = int(round(1/100 * (joint_max_x-joint_min_x)))
    expand_y = int(round(25/100 * (joint_max_y-joint_min_y)))

    if format == "cv2":
            
        image[max(0, joint_min_x - expand_x):min(joint_max_x + expand_x, image.shape[1])] = [0,0,0]
        
    elif format == "PIL":
            
        bottom = min(joint_max_y+expand_y, image.height)
        right = min(joint_max_x+expand_x,image.width) # +expand_x
        top = max(0,joint_min_y-expand_y)
        left = max(0,joint_min_x-expand_x) # -expand_x
        image = np.array(image)
            
        if len(image.shape) == 3:
            image[top:bottom,left:right] = [0,0,0]
        else:
            image[top:bottom,left:right] = np.min(image)
        
        return Image.fromarray(image)


class Video_dataset_cat_eval(Dataset):
    def __init__(self, data_pkl, transform=None, duration=16, audio=False, context=False, body=False):
        self.transform = transform
        self.duration = duration
        self.body = body
        self.context = context
        data_pickle = pickle.load(open(data_pkl, 'rb'))
        data_old = data_pickle['val_cat']
        self.openpose_path = '/gpu-data3/filby/aff-wild2/openpose/' 
        self.audio = audio

        with open('video_list.txt') as f:
            self.video_list = [line.rstrip() for line in f]
        
        with open('video_list.txt') as f:
            self.video_list_no_extension = [line.rstrip().replace('.mp4','').replace('.avi','') for line in f]

        self.audio = audio

        data_new = []

        for d in data_old:
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
            self.audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])
            self.mel_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop((112,112)),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.5], std=[0.5])]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # video frame list
        samples = self.data[idx].frames_list
        
        index = self.video_list_no_extension.index(str(self.data[idx].video_path.split('/')[-1]).replace('_right','').replace('_left',''))                
        
        # get video name with extension (.mp4/.avi)
        video_name = self.video_list[index]
        
        faces = []
        contexts = []
        raws = []
        expressions = []
        valid = []
        frame_ids = []

        if len(samples) < self.duration:
            for i in range(self.duration):
                
                frame_num = str(samples[i%len(samples)].frame_id + 1).zfill(5)                
                expression = samples[i%len(samples)].expression
                frame_ids.append(samples[i%len(samples)].frame_id)

                if self.context:
                    # compute raw frame path
                    raw_path = os.path.join(self.openpose_path, str(video_name) + '_frames', str(frame_num) + '.jpg')  
                    
                    # check if path to raw frame exists, if not get the previous one
                    while not os.path.exists(raw_path):
                        frame_num=str(int(frame_num)-1).zfill(5)
                        raw_path = os.path.join(self.openpose_path, str(video_name) + '_frames', str(frame_num) + '.jpg')
                    # get raw frame
                    raw = Image.open(raw_path).convert("RGB")
                
                    pos = 0
                
                    # compute bounding box and context
                    pose_path = os.path.join(self.openpose_path, str(video_name) + '_openpose_body25','json', str(frame_num) + '_keypoints.json')
                    # check if .json file exists, if not get raw frame
                    if (os.path.exists(pose_path)):    
                        pose = json.load(open(os.path.join(pose_path,)))
                        # check if .json file contains any data, if not get raw frame
                        if (len(pose['people']) != 0):
                            joints = pose['people'][pos]['pose_keypoints_2d']
                            context = get_context(raw, joints, format="PIL")
                        else:
                            context = raw
                    else:
                        context = raw
                        
                if self.body and not self.context:
                    raw_path = os.path.join(self.openpose_path, str(video_name) + '_frames', str(frame_num) + '.jpg')  
                    
                    # check if path to raw frame exists, if not get the previous one
                    while not os.path.exists(raw_path):
                        #print(raw_path)                    
                        frame_num=str(int(frame_num)-1).zfill(5)
                        raw_path = os.path.join(self.openpose_path, str(video_name) + '_frames', str(frame_num) + '.jpg')
                    # get raw frame
                    raw = Image.open(raw_path).convert("RGB")        
                
                # get aligned face crop
                face_path = samples[i%len(samples)].frame_path
                if face_path == 'not_detected':
                    face = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                else:
                    face = Image.open(face_path).convert("RGB")
                  
                if self.transform:
                    face = self.transform(face)
                    if self.context:
                        context = self.transform(context)
                    if self.body:
                        raw = self.transform(raw)
               
                faces.append(face)
                if self.context:
                    contexts.append(context)
                if self.body:
                    raws.append(raw)
                expressions.append(expression)
                valid.append(i<len(samples))
        else:
            for i in range(self.duration):
                frame_num = str(samples[i].frame_id + 1).zfill(5)                
                expression = samples[i].expression
                frame_ids.append(samples[i].frame_id)

                if self.context:
                    # compute raw frame path
                    raw_path = os.path.join(self.openpose_path, str(video_name) + '_frames', str(frame_num) + '.jpg')  
                    
                    # check if path to raw frame exists, if not get the previous one
                    while not os.path.exists(raw_path):
                        #print(raw_path)                    
                        frame_num=str(int(frame_num)-1).zfill(5)
                        raw_path = os.path.join(self.openpose_path, str(video_name) + '_frames', str(frame_num) + '.jpg')
                    # get raw frame
                    raw = Image.open(raw_path).convert("RGB")
                
                    pos = 0
                
                    # compute bounding box and context
                    pose_path = os.path.join(self.openpose_path, str(video_name) + '_openpose_body25','json', str(frame_num) + '_keypoints.json')
                    # check if .json file exists, if not get raw frame
                    if (os.path.exists(pose_path)):    
                        pose = json.load(open(os.path.join(pose_path,)))
                        # check if .json file contains any data, if not get raw frame
                        if (len(pose['people']) != 0):
                            joints = pose['people'][pos]['pose_keypoints_2d']
                            context = get_context(raw, joints, format="PIL")
                        else:
                            context = raw
                    else:
                        context = raw
                        
                if self.body and not self.context:
                    raw_path = os.path.join(self.openpose_path, str(video_name) + '_frames', str(frame_num) + '.jpg')  
                    
                    # check if path to raw frame exists, if not get the previous one
                    while not os.path.exists(raw_path):
                        #print(raw_path)                    
                        frame_num=str(int(frame_num)-1).zfill(5)
                        raw_path = os.path.join(self.openpose_path, str(video_name) + '_frames', str(frame_num) + '.jpg')
                    # get raw frame
                    raw = Image.open(raw_path).convert("RGB")        
                
                # get aligned face crop
                face_path = samples[i].frame_path
                if face_path == 'not_detected':
                    face = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                else:
                    face = Image.open(face_path).convert("RGB")
                  
                if self.transform:
                    face = self.transform(face)
                    if self.context:
                        context = self.transform(context)
                    if self.body:
                        raw = self.transform(raw)
               
                faces.append(face)
                if self.context:
                    contexts.append(context)
                if self.body:
                    raws.append(raw)
                expressions.append(expression)
                valid.append(True)   

        faces = torch.stack(faces)
        if self.context:
            contexts = torch.stack(contexts)
        if self.body:
            raws = torch.stack(raws)
        expressions = torch.LongTensor(expressions)
        valid = torch.BoolTensor(valid)

        # ---------------------- handle audio -------------------------- #
        if self.audio:
            fps = self.data[idx].video_fps

            audio_file = "/gpu-data3/filby/aff-wild2/wav_tnt/" + self.data[idx].video_path.split("/")[-1].split(".")[
                0] + ".wav"
            audio_file = audio_file.replace("_right","").replace("_left","")


            audio_frames = []


            for i in frame_ids:
                # print(i)
                frame_offset = i/fps*44100 - 5 * 44100
                if frame_offset < 0:
                    frame_offset = 0

                num_frames = 10*44100

                audioData, sampleRate = torchaudio.load(audio_file, offset=int(frame_offset), num_frames=int(num_frames))

                assert sampleRate == 44100

                audio = self.audio_spec_transform(self.audio_transform(audioData))

                if audio.size(2) != 1001:
                    audio = F.pad(audio,[0, 1001-audio.size(2)])


                # audio = self.mel_transform(audio_img)
                audio_frames.append(audio)

        if self.context and self.body and self.audio:
            return faces, raws, contexts, expressions, valid, torch.stack(audio_frames)
        elif self.context and self.body and not self.audio:
            return faces, raws, contexts, expressions, valid        
        elif self.context and not self.body and not self.audio:
            return faces, contexts, expressions, valid        
        elif not self.context and self.body and not self.audio:
            return faces, raws, expressions, valid
        elif not self.context and not self.body and self.audio:
            return faces, expressions, valid, torch.stack(audio_frames)
        else:
            return faces, expressions, valid


class Video_dataset_cont_eval(Dataset):
    def __init__(self, data_pkl, transform=None, duration=16):
        self.transform = transform
        self.duration = duration
        data_pickle = pickle.load(open(data_pkl, 'rb'))
        data_old = data_pickle['val_cont']
        
        data_new = []
        for d in data_old:
            frames = d.frames_list
            for i in range(0, len(frames), self.duration):
                end = min(i+self.duration, len(frames))
                data_new.append(Video_annotation(video_path=d.video_path, frames_list=frames[i:end]))
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
                image = Image.open(samples[i%len(samples)].frame_path).convert("RGB")
                val = samples[i%len(samples)].valence
                ar = samples[i%len(samples)].arousal
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
                valence.append(val)
                arousal.append(ar)
                valid.append(i<len(samples))
        else:
            for i in range(self.duration):
                image = Image.open(samples[i].frame_path).convert("RGB")
                val = samples[i].valence
                ar = samples[i].arousal
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
                valence.append(val)
                arousal.append(ar)
                valid.append(True)

        frames = torch.stack(frames)
        valence = torch.FloatTensor(valence)
        arousal = torch.FloatTensor(arousal)
        cont = torch.stack([valence, arousal], dim=1)
        valid = torch.BoolTensor(valid)
        return frames, cont, valid


