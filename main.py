import time

import numpy as np
from scipy.io.wavfile import read as read_wav

import analyze
import record
import model
import user_commands
import audio_conv


def main():
    record_voice()

def record_voice(filename=None):
    frames = record.record_frames()
    if filename is None:
        record.write_frames(frames)
    else:
        record.write_frames(frames, filename)

if __name__ == "__main__":
    main()
