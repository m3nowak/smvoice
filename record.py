import wave

import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 22050
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "output.wav"


def record_frames():
    """
    Record audio frames
        :return list: recorded frames
    """

    pa = pyaudio.PyAudio()

    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)

    print("Recording started")

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    return frames


def write_frames(frames, filename=WAVE_OUTPUT_FILENAME):
    """
    Record audio frames
        :frames list: list of recorded frames
        :filename str: name of saved file
    """
    pa = pyaudio.PyAudio()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
