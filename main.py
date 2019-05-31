import time

import numpy as np
from scipy.io.wavfile import read as read_wav

import analyze
import record
import model


def main():
    feat, label = model.create_training_info('samples_train.csv')
    clsfr = model.create_clsfr()
    model.train_clsfr(clsfr, feat, label)
    feat_test, label_test = model.create_training_info('samples_test.csv')
    score = model.score_clsfr(clsfr, feat_test, label_test)
    #freq, wf = read_wav('output.wav')

    #features = analyze.extract_features(freq, wf)

    print(score)
    #frames = record.record_frames()
    # print(type(frames))
    # print(type(frames[0]))
    # record.write_frames(frames, 'badsample8.wav')


if __name__ == "__main__":
    main()
