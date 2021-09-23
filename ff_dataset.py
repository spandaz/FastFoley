import random
r = random.random
import json
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from torch.utils.data import Dataset
import os
import torch
import torch.nn as nn
from tqdm import tqdm


class videoData(Dataset):
    def __init__(self, video_feature_path, frameNummax=860, phase="train"):
        super(videoData, self).__init__()
        self.data_path = video_feature_path
        self.frameNummax = frameNummax
        self.video_feature_list = []
        self.label_list = []
        self.wav_feature = []
        self.wav_feature_mean = []
        self.video_dir = os.listdir(video_feature_path)

        self.video_name_list = []

        for i in self.video_dir:
            for j in os.listdir('%s/%s' % (video_feature_path, i)):
                self.video_feature_list.append('%s/%s/%s' % (video_feature_path, i, j))
                self.label_list.append(j.split('_')[0])
                self.wav_feature.append('wavfeature_5cls/%s/%s.npy'%(i, j.split('.')[0]))
                self.wav_feature_mean.append('wavfeature_5cls/%s_mean.npy' % (i))

                self.video_name_list.append(j)
        
        # shuffle
        random.seed(0)
        random.shuffle(self.video_feature_list, random=r)
        random.seed(0)
        random.shuffle(self.label_list, random=r)
        random.seed(0)
        random.shuffle(self.wav_feature, random=r)
        random.seed(0)
        random.shuffle(self.wav_feature_mean, random=r)
        random.seed(0)
        random.shuffle(self.video_name_list, random=r)

        self.classname = np.unique(self.label_list)
        self.frameNummax = frameNummax

        with open(os.path.join("./", "cls.json")) as f:
            self.cls_map = json.load(f)

        total_size = len(self.label_list)
        train_size = int(total_size * 0.8)
        val_size = total_size - train_size
        if phase == "train":
            self.video_feature_list = self.video_feature_list[:train_size]
            self.label_list = self.label_list[:train_size]
            self.wav_feature = self.wav_feature[:train_size]
            self.wav_feature_mean = self.wav_feature_mean[:train_size]
            self.video_name_list = self.video_name_list[:train_size]
        elif phase == "val":
            self.video_feature_list = self.video_feature_list[train_size:]
            self.label_list = self.label_list[train_size:]
            self.wav_feature = self.wav_feature[train_size:]
            self.wav_feature_mean = self.wav_feature_mean[train_size:]
            self.video_name_list = self.video_name_list[train_size:]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):

        video_feature = np.load(self.video_feature_list[index])[:self.frameNummax]

        label = np.where(self.classname == self.label_list[index])[0]

        # category_embedding = torch.nn.functional.one_hot(label, n_category)
        cls = self.label_list[index]
        cls_id = self.cls_map[cls]

        wavfeature_ = np.load(self.wav_feature[index])
        wavfeature_mean_ = np.load(self.wav_feature_mean[index])

        wavfeature_res = wavfeature_ - wavfeature_mean_

        feature_len = self.frameNummax

        return [index, video_feature, feature_len, wavfeature_, wavfeature_res, wavfeature_mean_, label, cls_id]

# data = videoData('video',50)
#
# dataloader = torch.utils.data.DataLoader(data,batch_size=2,shuffle=True)
#
# for index,spimage,first,label in dataloader:
#     print(index)
#     break


# for index in range(len(videolist)):
#     vc = cv2.VideoCapture(videolist[index])  # 读入视频文件
#     # frameNum = int(vc.get(7))  # 总帧数，等帧率*时长
#     spimage = []
#     for i in range(0, 3):
#         vc.set(cv2.CAP_PROP_POS_FRAMES, i)
#         rval, frame = vc.read()
#         sp = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         if i == 0:
#             first = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#         # frame = cv2.resize(frame, (80, 80))
#         spimage.append(sp)
#     spimage = np.transpose(spimage,[1,2,0])
#     label = np.where(classname==labellist[index])