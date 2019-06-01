import time

import numpy as np
from scipy.io.wavfile import read as read_wav

import analyze
import record
import model
import user_commands
import audio_conv

import gmm_model

def main():
    gmm_test_run()

def gmm_test_run():
    gm = gmm_model.train_model()
    test_res = gmm_model.test_model(gm)
    for test_re in test_res:
        print(test_re)

def record_voice(filename=None, length_sec=3):
    frames = record.record_frames(length_sec)
    if filename is None:
        record.write_frames(frames)
    else:
        record.write_frames(frames, filename)

if __name__ == "__main__":
    main()
