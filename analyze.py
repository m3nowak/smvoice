import math

import python_speech_features as psf
import numpy as np
from scipy.io.wavfile import read as read_wav
import sklearn.preprocessing as skpp

import record

WIN_LEN = 0.025


def open_wave(filename):
    return read_wav(filename)

def best_nftt(frequency, window_length):
    minimal_value = frequency*window_length
    power_determined = math.ceil(math.log2(minimal_value))
    return int(math.pow(2, power_determined))


def extract_features(freq, wf):
    nftt = best_nftt(freq, WIN_LEN)
    mfcc_feat = psf.mfcc(wf, freq, winlen=WIN_LEN,
                         winstep=0.01, numcep=13, nfft=nftt, winfunc=np.bartlett)
    mfcc_feat_scaled = skpp.scale(mfcc_feat)
    mfcc_feat_d1 = psf.delta(mfcc_feat_scaled, 3)
    mfcc_feat_d2 = psf.delta(mfcc_feat_d1, 3)

    return np.hstack((mfcc_feat_scaled, mfcc_feat_d1, mfcc_feat_d2))

def extract_features_file(filename):
    freq, wf = open_wave(filename)
    feat = extract_features(freq, wf)
    return feat