import torch
import os
import yaml
from torchvision import models
from ff_dataset import videoData
from model.fastfoley import FastSpeech2
from FSLSTM import LSTM
import librosa
import librosa.display
import soundfile as sf
import gl
import numpy as np
# from wavenet_vocoder import builder
# from TransConv import TransConv
# from sklearn.model_selection import train_test_split
import tqdm

device = torch.device('cuda')
val_dataset = videoData('./video_feature_5cls', 860, "val")
test_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)


'''
def build_wavenet(checkpoint_path=None, device='cuda:0'):
    model = builder.wavenet(
        out_channels=30,
        layers=24,
        stacks=4,
        residual_channels=512,
        gate_channels=512,
        skip_out_channels=256,
        cin_channels=80,
        gin_channels=-1,
        weight_normalization=True,
        n_speakers=None,
        dropout=0.05,
        kernel_size=3,
        upsample_conditional_features=True,
        upsample_scales=[4, 4, 4, 4],
        freq_axis_kernel_size=3,
        scalar_input=True,
    )

    model = model.to(device)
    if checkpoint_path:
        print("Load WaveNet checkpoint from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.make_generation_fast_()

    return model

def gen_waveform(model, save_path, c, device):
    initial_input = torch.zeros(1, 1, 1).to(device)
    if c.shape[1] != 80: # mel_channels
        c = np.swapaxes(c, 0, 1)
    length = c.shape[0] * 256
    c = torch.FloatTensor(c.T).unsqueeze(0).to(device)
    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=None, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=np.log(1e-14))
    waveform = y_hat.view(-1).cpu().data.numpy()
    # librosa.output.write_wav(save_path, waveform, sr=22050)
    sf.write(save_path, waveform, samplerate=44100, subtype='PCM_24')

# vocoder
wavenet_model = build_wavenet('./hammer_checkpoint_step000137000_ema.pth', device)
'''
model_config = yaml.load(open('./config/model.yaml', "r"), Loader=yaml.FullLoader)

FastFoley = FastSpeech2(model_config)
FastFoley = FastFoley.to(device)

checkpoint_FastFoley = torch.load('/ceph/home/lsp20/FastFoley/ckpt/ckpt_5cls/ckpt_3000.pt', map_location=torch.device('cuda'))
FastFoley.load_state_dict(checkpoint_FastFoley, False)
FastFoley.eval()
# tc = TransConv()
# tc = tc.to(device)
# checkpoint_tc = torch.load('../checkpoint_199_tc.pt', map_location=torch.device('cuda'))
# tc.load_state_dict(checkpoint_tc, False)
# tc.eval()

print("testing..")

for index, feature_batch, feature_batch_len, wavfeature_, wavfeature_res, wavfeature_mean_, gt_label, cls_id in tqdm.tqdm(test_dataloader):
    index, feature_batch, feature_batch_len, wavfeature_, wavfeature_res, wavfeature_mean_, gt_label, cls_id = \
        index.cuda(), feature_batch.cuda(), feature_batch_len.cuda(), wavfeature_.cuda(), wavfeature_res.cuda(), \
        wavfeature_mean_.cuda(), gt_label.cuda(), cls_id.cuda()

    # feature_batch = tc(feature_batch)
    feature_batch = feature_batch.float()
    wavfeature_ = wavfeature_.float()

    output, postnet_output = FastFoley(cls_id, feature_batch, feature_batch_len, 860, wavfeature_, feature_batch_len, 860)

    postnet_output = postnet_output.permute([0, 2, 1])
    output = output.permute([0, 2, 1])

    # wavfeature_ = wavfeature_.permute([0, 2, 1])
    # wavfeature_res = wavfeature_res.permute([0, 2, 1])
    # wavfeature_mean_ = wavfeature_mean_.permute([0, 2, 1])
    #
    # wavfeature_res = wavfeature_res.float()
    # wavfeature_mean_ = wavfeature_mean_.float()
    # wavfeature_ = wavfeature_.float()

    Spectrogram = postnet_output + wavfeature_mean_
    Spectrogram_ = output + wavfeature_mean_
    # Spectrogram = sc
    #
    Spectrogram = Spectrogram.cpu()
    Spectrogram = Spectrogram.squeeze()

    video_name = val_dataset.video_name_list[index]
    # save_path = './savewav/test_%s.wav' % video_name
    # gen_waveform(wavenet_model, save_path, Spectrogram, device)
    # wav = gl._griffin_lim(Spectrogram.detach().numpy()[:, :])
    # wav = gl.save_wav(wav, './savewav/test_%s.wav' % video_name[:-4], 44100)

    for j in range(0, len(index)):
        video_name = val_dataset.video_name_list[index[j]]
        Spectrogram = Spectrogram.cpu()
        Spectrogram = Spectrogram.squeeze()
        # print('Spectrogram.shape:', Spectrogram.shape)
        # Spectrogram_all = Spectrogram_all.to(device)
        # break
        save_path = './testwav/5cls/test_%s.wav' % video_name
        # gen_waveform(wavenet_model, save_path, Spectrogram, device)
        wav = gl._griffin_lim(Spectrogram.detach().numpy()[:, :])
        wav = gl.save_wav(wav, './testwav/5cls/test_%s.wav' % video_name[:-4], 44100)