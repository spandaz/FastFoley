from torchvision import models
import torch
import os
import numpy as np
import cv2
import tqdm

videopath = 'data10s'
savedir = 'video_feature'
frameNummax = 213

device = torch.device('cuda')

resnet50 = models.resnet50(pretrained=True)
resnet50.fc = torch.nn.Sequential()
resnet50.to(device)
resnet50.eval()

datapath = videopath
videolist = []
videodir = os.listdir(videopath)

for i in videodir:
    for j in os.listdir('%s/%s' % (videopath, i)):
        videolist.append('%s/%s/%s' % (videopath, i, j))

for index in tqdm.tqdm(range(419, len(videolist))):
    vc = cv2.VideoCapture(videolist[index])  # 读入视频文件

    video_feature = []
    first = None

    spimage_list = []
    first_list = []

    for f in range(0, frameNummax):
        spimage = []

        for i in range(f, f+3):
            vc.set(cv2.CAP_PROP_POS_FRAMES, i)
            rval, frame = vc.read()
            #print(i, f)
            frame = cv2.resize(frame, (224, 224))
            sp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if i == f:
                first = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                first = np.transpose(first,[2,0,1])  # to C,H,W
            spimage.append(sp)

        spimage = np.array(spimage).astype("float") / 255.#.unsqueeze(dim=0)
        first = np.array(first).astype("float") / 255.#.unsqueeze(dim=0)

        spimage_list.append(spimage)
        first_list.append(first)

    spimage_list = np.array(spimage_list)
    first_list = np.array(first_list)

    infer_batch = 256
    if frameNummax % infer_batch == 0:
        num = frameNummax // infer_batch
    else:
        num = frameNummax // infer_batch + 1
    for i in range(num):
        spimage_batch = spimage_list[i * infer_batch: (i + 1) * infer_batch]
        first_batch = first_list[i * infer_batch: (i + 1) * infer_batch]

        spimage_batch = torch.tensor(spimage_batch).float().to(device)
        first_batch = torch.tensor(first_batch).float().to(device)

        with torch.no_grad():
            feature1 = resnet50(spimage_batch)
            feature2 = resnet50(first_batch)
            feature_cat = torch.cat([feature1, feature2], dim=1)

        video_feature.append(feature_cat.detach().cpu())

    video_feature = torch.cat(video_feature, dim=0)

    video_feature = np.array(video_feature)

    os.makedirs(os.path.join(savedir, videolist[index].split("/")[1]), exist_ok=True)
    save_path = os.path.join(savedir, videolist[index].split("/")[1], videolist[index].split("/")[-1][:-4]) + ".npy"
    np.save(save_path, video_feature)