import time

import numpy as np
from scipy.io.wavfile import read as read_wav

import analyze
import record
import model
import user_commands


def main():
    #user_commands.make_clsfr()
    user_commands.perform_live_test()
    #user_commands.test_trial()
    #feat, label = model.create_training_info('samples_train.csv')
    #clsfr = model.create_clsfr()
    #model.train_clsfr(clsfr, feat, label)
    #feat_test, label_test = model.create_training_info('samples_test.csv')
    #score = model.determine(clsfr, feat_test)
    #freq, wf = read_wav('output.wav')

    #features = analyze.extract_features(freq, wf)

    #print(score)
    #frames = record.record_frames()
    # print(type(frames))
    # print(type(frames[0]))
    #record.write_frames(frames, 'sample6.wav')

def record_voice(filename=None):
    frames = record.record_frames()
    if filename is None:
        record.write_frames(frames)
    else:
        record.write_frames(frames, filename)

if __name__ == "__main__":
    main()
