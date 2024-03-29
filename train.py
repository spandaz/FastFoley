import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchvision import models
from ff_dataset import videoData
import librosa
# import gl
import numpy as np
# from TransConv import TransConv
import tqdm
import tensorboardX

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
# from dataset import Dataset

from evaluate import evaluate

from model.fastfoley import FastSpeech2

device = torch.device('cuda')

train_dataset = videoData('./video_feature', 860, "train")
val_dataset = videoData('./video_feature', 860, "val")

model_config = yaml.load(open('./config/model.yaml', "r"), Loader=yaml.FullLoader)

FastFoley = FastSpeech2(model_config)
FastFoley = FastFoley.to(device)

batch_size = 16
batch_size_per = 16


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

print("Initializing distributed training")

opt = torch.optim.Adam([
    {'params': FastFoley.parameters(), 'lr': 1e-3},
])

_loss = torch.nn.SmoothL1Loss()

train_iter = 0
val_iter = 0
summary = tensorboardX.SummaryWriter()

for epoch in range(500):

    # Train
    s = 0
    # loss_value = 0
    spectro_loss_value = 0

    train_bar = tqdm.tqdm(train_dataloader)
    for index_all, feature_batch_all, feature_batch_len_all, wavfeature_all, wavfeature_res_all, wavfeature_mean_all, gt_label_all in train_bar:

        Spectrogram_all = []
        # print('feature_batch_all.shape:', feature_batch_all.shape)
        if batch_size % batch_size_per == 0:
            num = batch_size // batch_size_per
        else:
            num = batch_size // batch_size_per + 1

        loss_batch = 0
        for i in range(num):

            index = index_all[i * batch_size_per: (i + 1) * batch_size_per]
            feature_batch = feature_batch_all[i * batch_size_per: (i + 1) * batch_size_per]
            feature_batch_len = feature_batch_len_all[i * batch_size_per: (i + 1) * batch_size_per]
            wavfeature_ = wavfeature_all[i * batch_size_per: (i + 1) * batch_size_per]
            wavfeature_res = wavfeature_res_all[i * batch_size_per: (i + 1) * batch_size_per]
            wavfeature_mean_ = wavfeature_mean_all[i * batch_size_per: (i + 1) * batch_size_per]

            gt_label = gt_label_all[i * batch_size_per: (i + 1) * batch_size_per]

            # 将整个 batch_size 分为 num 次计算
            index, feature_batch, feature_batch_len, wavfeature_, wavfeature_res, wavfeature_mean_, gt_label = \
            index.cuda(), feature_batch.cuda(), feature_batch_len.cuda(), wavfeature_.cuda(), wavfeature_res.cuda(), \
            wavfeature_mean_.cuda(), gt_label.cuda()

            # feature_batch = tc(feature_batch)
            # feature_batch = feature_batch.permute([0, 2, 1])

            feature_batch = feature_batch.float()

            output, postnet_output = FastFoley(feature_batch, feature_batch_len, 860, None, None, None)
            # sc = sc.permute([0, 2, 1])
            postnet_output = postnet_output.permute([0, 2, 1])

            wavfeature_mean_ = wavfeature_mean_.float()
            Spectrogram = postnet_output + wavfeature_mean_
            #
            wavfeature_ = wavfeature_.float()
            # print(postnet_output.dtype)
            spectro_loss = _loss(Spectrogram, wavfeature_) * len(index)
            spectro_loss_value = spectro_loss_value + spectro_loss.item()
            loss_batch = loss_batch + spectro_loss.item()

            # 只求梯度，每次都会叠加
            spectro_loss.backward()

            Spectrogram_all.append(Spectrogram)

        opt.step()
        opt.zero_grad()
        s = s + len(index_all)

        summary.add_scalar(tag="train loss", scalar_value=loss_batch / len(index_all), global_step=train_iter)

        train_bar.set_description("[Train %d], spectro_loss=%f" % (epoch + 1, spectro_loss_value / s))

        # if train_iter % 100 == 1:
        #     Spectrogram_all = torch.cat(Spectrogram_all, dim=0)
        #     for i in range(0, len(index_all)):
        #         video_name = train_dataset.video_name_list[index_all[i]]
        #         Spectrogram_all = Spectrogram_all.cpu()
        #         wav = gl._griffin_lim(Spectrogram_all.detach().numpy()[i, :, :])
        #         wav = gl.save_wav(wav, './savewav/train_%s.wav' % video_name[:-4], 44100)

        train_iter += 1

    # Validate
    s = 0
    # loss_value = 0
    spectro_loss_value = 0

    val_bar = tqdm.tqdm(val_dataloader)
    for index_all, feature_batch_all, feature_batch_len_all, wavfeature_all, wavfeature_res_all, wavfeature_mean_all, gt_label_all in val_bar:

        Spectrogram_all = []

        if batch_size % batch_size_per == 0:
            num = batch_size // batch_size_per
        else:
            num = batch_size // batch_size_per + 1

        with torch.no_grad():
            loss_batch = 0
            for i in range(num):

                index = index_all[i * batch_size_per: (i + 1) * batch_size_per]
                feature_batch = feature_batch_all[i * batch_size_per: (i + 1) * batch_size_per]
                feature_batch_len = feature_batch_len_all[i * batch_size_per: (i + 1) * batch_size_per]
                wavfeature_ = wavfeature_all[i * batch_size_per: (i + 1) * batch_size_per]
                wavfeature_res = wavfeature_res_all[i * batch_size_per: (i + 1) * batch_size_per]
                wavfeature_mean_ = wavfeature_mean_all[i * batch_size_per: (i + 1) * batch_size_per]

                gt_label = gt_label_all[i * batch_size_per: (i + 1) * batch_size_per]

                # 将整个 batch_size 分为 num 次计算
                index, feature_batch, feature_batch_len, wavfeature_, wavfeature_res, wavfeature_mean_, gt_label = \
                    index.cuda(), feature_batch.cuda(), feature_batch_len.cuda(), wavfeature_.cuda(), wavfeature_res.cuda(), \
                    wavfeature_mean_.cuda(), gt_label.cuda()

                # feature_batch = tc(feature_batch)
                # feature_batch = feature_batch.permute([0, 2, 1])

                feature_batch = feature_batch.float()

                output, postnet_output = FastFoley(feature_batch, feature_batch_len, 860, None, None, None)
                
                # sc = sc.permute([0, 2, 1])
                postnet_output = postnet_output.permute([0, 2, 1])
                wavfeature_mean_ = wavfeature_mean_.float()
                Spectrogram = postnet_output + wavfeature_mean_
                #
                wavfeature_ = wavfeature_.float()
                spectro_loss = _loss(Spectrogram, wavfeature_) * len(index)
                spectro_loss_value = spectro_loss_value + spectro_loss.item()
                loss_batch = loss_batch + spectro_loss.item()

                Spectrogram_all.append(Spectrogram)

            s = s + len(index_all)

        summary.add_scalar(tag="validate loss", scalar_value=loss_batch / len(index_all), global_step=val_iter)

        val_bar.set_description("[Val %d], spectro_loss=%f" % (epoch + 1, spectro_loss_value / s))

        # if val_iter % 100 == 1:
        #     Spectrogram_all = torch.cat(Spectrogram_all, dim=0)
        #     for i in range(0, len(index_all)):
        #         video_name = val_dataset.video_name_list[index_all[i]]
        #         Spectrogram_all = Spectrogram_all.cpu()
        #         wav = gl._griffin_lim(Spectrogram_all.detach().numpy()[i, :, :])
        #         wav = gl.save_wav(wav, './savewav/val_%s.wav' % video_name[:-4], 44100)

        val_iter += 1

    if (epoch + 1) % 100 == 0:

        torch.save(FastFoley.state_dict(), './ckpt/ckpt%s.pt' % (epoch + 1))
        # torch.save(opt.state_dict(),'./ckpt/ckpt.opt_%s.pt' % (epoch + 1))
        #     {
        #         "FastFoley": FastFoley.state_dict(),
        #         "optimizer": opt._optimizer.state_dict(),
        #     })
