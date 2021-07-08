import os
import cv2
import json
import math
import torch
import pickle
import random
import joblib
import torchaudio
import numpy as np
import torchvision
import torchvision.transforms
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as tF

from PIL import Image
from torch.utils.data import Dataset

from .dataset import Video_annotation, Frame_annotation_cat, Frame_annotation_cont


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


def get_bounding_box(image, joints, format="PIL"):
    joints = np.reshape(joints, (25, 3))
    joints[joints[:,2]<0.1] = np.nan
    joints[np.isnan(joints[:,2])] = np.nan

    joint_min_x = int(round(np.nanmin(joints[:,0])))
    joint_min_y = int(round(np.nanmin(joints[:,1])))

    joint_max_x = int(round(np.nanmax(joints[:,0])))
    joint_max_y = int(round(np.nanmax(joints[:,1])))

    expand_x = int(round(100/100 * (joint_max_x-joint_min_x)))
    expand_y = int(round(100/100 * (joint_max_y-joint_min_y)))

    if format == "cv2":
        return image[max(0,joint_min_y-expand_y):min(joint_max_y+expand_y, image.shape[0]), max(0,joint_min_x-expand_x):min(joint_max_x+expand_x,image.shape[1])]
    elif format == "PIL":
        bottom = min(joint_max_y+expand_y, image.height)
        right = min(joint_max_x+expand_x,image.width)
        top = max(0,joint_min_y-expand_y)
        left = max(0,joint_min_x-expand_x)
        return tF.crop(image, top, left, bottom-top ,right-left)


class Video_dataset_cat(Dataset):
    def __init__(self, data_pkl, train=True, transform=None, duration=16, audio=False, audio_transform=None, context=False, body=False):
        
        self.train = train
        self.transform = transform
        self.duration = duration
        self.audio = audio
        self.audio_transform = audio_transform
        self.context = context
        self.body = body
        self.audio = audio
        self.openpose_path = '/gpu-data3/filby/aff-wild2/openpose/'
        self.embeddings = np.load("abaw_expr_embeddings.npy")

        data_pickle = pickle.load(open(data_pkl, 'rb'))
        
        with open('video_list.txt') as f:
            self.video_list = [line.rstrip() for line in f]
        
        with open('video_list.txt') as f:
            self.video_list_no_extension = [line.rstrip().replace('.mp4','').replace('.avi','') for line in f]
        
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

        
        if self.audio:

            t = []

            import torchaudio.transforms as at

            self.audio_transform = transforms.Compose(t)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
        valence = []
        arousal = []

        for i in range(self.duration):
            f_i = min(len(samples)-1,i)
            valid.append(i < len(samples))

            frame_ids.append(samples[f_i].frame_id)

            frame_num = str(samples[f_i].frame_id + 1).zfill(5)
            expression = samples[f_i].expression
            val = 0# samples[f_i].valence
            ar = 0 #samples[f_i].arousal

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

            if not self.audio:
                # get aligned face crop
                face_path = samples[f_i].frame_path
                if face_path == 'not_detected':
                    face = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                else:
                    face = Image.open(face_path).convert("RGB")            


            if self.transform:
                if not self.audio:
                    face = self.transform(face)
                if self.context:
                    context = self.transform(context)
                if self.body:
                    raw = self.transform(raw)

            if not self.audio:
                faces.append(face)
            if self.context:
                contexts.append(context)
            if self.body:
                raws.append(raw)

            expressions.append(expression)
            valence.append(val)
            arousal.append(ar)

        if not self.audio:
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

            mel_file = "/gpu-data3/filby/aff-wild2/wav_melspec/" + self.data[idx].video_path.split("/")[-1].split(".")[0] + ".npy"
            mel_file = mel_file.replace("_right","").replace("_left","")
            mel = np.load(mel_file)

            # print(np.max(mfc),np.min(mfc))
            audio_frames = []

            window_stride = 512/44100 # seconds

            # mel spec
            for i in frame_ids:
                seconds_video_offset = i/fps - 5
                if seconds_video_offset < 0:
                    seconds_video_offset = 0

                frame_offset = int(seconds_video_offset/window_stride)

                l = int(10/window_stride)
                audio_features = mel[:,:,frame_offset:frame_offset+l]

                # audio_features = self.scaler.transform(audio_features.T).T

                audio_features = torch.Tensor(audio_features)#.unsqueeze(0)
                # print(audio_features.size())
                if audio_features.size(2) != l:
                    audio_features = F.pad(audio_features,[0, l-audio_features.size(2)])


                audio_features = self.audio_transform(audio_features)

                audio_frames.append(audio_features)

            audio = torch.stack(audio_frames)
            
        data_dic = {}
        if self.context:
            data_dic['context'] = contexts
        if self.body:
            data_dic['body'] = raws
        if self.audio:
            data_dic['audio'] = audio

        data_dic['valid'] = valid
        if not self.audio:
            data_dic['faces'] = faces
        data_dic['expressions'] = expressions

        data_dic['valence'] = valence

        data_dic['embeddings'] = torch.tensor(self.embeddings).float()

        data_dic['arousal'] = arousal

        return data_dic


class Video_dataset_cont(Dataset):
    def __init__(self, data_pkl, train=True, transform=None, duration=16, audio=False, audio_transform=None, context=False, body=False):

        self.train = train
        self.transform = transform
        self.duration = duration
        self.audio = audio
        self.audio_transform = audio_transform
        self.context = context
        self.body = body
        self.audio = audio
        self.openpose_path = '/gpu-data3/filby/aff-wild2/openpose/'

        data_pickle = pickle.load(open(data_pkl, 'rb'))

        with open('video_list.txt') as f:
            self.video_list = [line.rstrip() for line in f]

        with open('video_list.txt') as f:
            self.video_list_no_extension = [line.rstrip().replace('.mp4', '').replace('.avi', '') for line in f]

        if train:
            self.data = data_pickle['train_cont']
        else:
            self.data = data_pickle['val_cont']


        data_new = []
        for d in self.data:
            frames = d.frames_list
            for i in range(0, len(frames), self.duration):
                end = min(i + self.duration, len(frames))
                data_new.append(Video_annotation(video_path=d.video_path, frames_list=frames[i:end], fps=d.video_fps))

        self.data = data_new

        if self.audio:
            t = []
            import torchaudio.transforms as at

            t.append(at.TimeMasking(100))

            t.append(at.FrequencyMasking(10))

            self.audio_transform = transforms.Compose(t)

            self.scaler = joblib.load("affwild2/preprocess_stuff/minmax_audio_scaler.pkl")

            self.audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[
                19.895])])  # , torchaudio.transforms.TimeMasking(100), torchaudio.transforms.FrequencyMasking(10)])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        samples = self.data[idx].frames_list

        index = self.video_list_no_extension.index(
            str(self.data[idx].video_path.split('/')[-1]).replace('_right', '').replace('_left', ''))

        faces = []
        contexts = []
        raws = []
        expressions = []
        valid = []
        frame_ids = []
        valence = []
        arousal = []

        for i in range(self.duration):
    
            f_i = min(len(samples) - 1, i)
            valid.append(i < len(samples))

            frame_ids.append(samples[f_i].frame_id)

            frame_num = str(samples[f_i].frame_id + 1).zfill(5)
            expression = 0 # samples[f_i].expression
            val = samples[f_i].valence
            ar = samples[f_i].arousal

            if self.context:
                # compute raw frame path
                raw_path = os.path.join(self.openpose_path, str(video_name) + '_frames', str(frame_num) + '.jpg')

                # check if path to raw frame exists, if not get the previous one
                while not os.path.exists(raw_path):
                    # print(raw_path)
                    frame_num = str(int(frame_num) - 1).zfill(5)
                    raw_path = os.path.join(self.openpose_path, str(video_name) + '_frames', str(frame_num) + '.jpg')
                # get raw frame
                raw = Image.open(raw_path).convert("RGB")

                pos = 0

                # compute bounding box and context
                pose_path = os.path.join(self.openpose_path, str(video_name) + '_openpose_body25', 'json',
                                         str(frame_num) + '_keypoints.json')
                # check if .json file exists, if not get raw frame
                if (os.path.exists(pose_path)):
                    pose = json.load(open(os.path.join(pose_path, )))
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
                    # print(raw_path)
                    frame_num = str(int(frame_num) - 1).zfill(5)
                    raw_path = os.path.join(self.openpose_path, str(video_name) + '_frames', str(frame_num) + '.jpg')
                # get raw frame
                raw = Image.open(raw_path).convert("RGB")

            if not self.audio:
                # get aligned face crop
                face_path = samples[f_i].frame_path
                if face_path == 'not_detected':
                    face = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8), 'RGB')
                else:
                    face = Image.open(face_path).convert("RGB")


            if self.transform:
                if not self.audio:
                    face = self.transform(face)
                if self.context:
                    context = self.transform(context)
                if self.body:
                    raw = self.transform(raw)
                if self.optical_flow:
                    flow_x = self.optical_flow_transform(flow_x)
                    flow_y = self.optical_flow_transform(flow_y)

            if not self.audio:
                faces.append(face)
            if self.context:
                contexts.append(context)
            if self.body:
                raws.append(raw)

            expressions.append(expression)
            valence.append(val)
            arousal.append(ar)

        if not self.audio:
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

                audio_features = self.audio_transform(audio_features)

                audio_frames.append(audio_features)

            audio = torch.stack(audio_frames)

        data_dic = {}
        if self.context:
            data_dic['context'] = contexts
        if self.body:
            data_dic['body'] = raws
        if self.audio:
            data_dic['audio'] = audio

        data_dic['valid'] = valid
        if not self.audio:
            data_dic['faces'] = faces
        data_dic['expressions'] = expressions

        valence = torch.FloatTensor(valence)
        arousal = torch.FloatTensor(arousal)
        data_dic['valence'] = valence

        data_dic['arousal'] = arousal
        # print(valence,arousal)
        data_dic['cont'] = torch.stack([valence, arousal], dim=1)

        return data_dic

