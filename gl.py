import numpy as np
from scipy.io import wavfile
import librosa

gl_iters = 60
n_fft, hop_length, win_length = 1024, None, None


def save_wav(wav, path, sr):

    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))


def _griffin_lim(S):  # S: Spectrogram

    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)

    for i in range(gl_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)

    return y


def _stft(y):

    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):

    return librosa.istft(y, hop_length=hop_length, win_length=win_length)
