import time

import numpy as np
from scipy.io.wavfile import read as read_wav

import analyze
import record


def main():
    freq, wf = read_wav('output.wav')

    features = analyze.extract_features(freq, wf)

    print('uuuh')
    #frames = record.record_frames()
    # print(type(frames))
    # print(type(frames[0]))
    # record.write_frames(frames)


if __name__ == "__main__":
    main()
