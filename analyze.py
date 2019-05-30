import math

import python_speech_features as psf
import numpy as np

import record

WIN_LEN = 0.025


def best_nftt(frequency, window_length):
    minimal_value = frequency*window_length
    power_determined = math.ceil(math.log2(minimal_value))
    return int(math.pow(2, power_determined))


def extract_features(freq, wf):
    nftt = best_nftt(freq, WIN_LEN)
    mfcc_feat = psf.mfcc(wf, freq, winlen=WIN_LEN,
                         winstep=0.01, numcep=13, nfft=nftt, winfunc=np.bartlett,
                         )
    mfcc_feat_d1 = psf.delta(mfcc_feat, 2)
    mfcc_feat_d2 = psf.delta(mfcc_feat_d1, 2)

    return np.hstack((mfcc_feat, mfcc_feat_d1, mfcc_feat_d2))
