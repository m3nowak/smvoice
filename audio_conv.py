import os

import librosa

MP3_DIR = 'artifacts/mp3s'
WAV_DIR = 'artifacts/wav_corp'

def convert_files():
    counter = 1
    for filename in os.listdir(MP3_DIR):
        y, sr = librosa.core.load(os.path.join(MP3_DIR,filename), sr=22050)
        target_filename = os.path.join(WAV_DIR, '{:03d}.wav'.format(counter))
        librosa.output.write_wav(target_filename,y,sr)
        counter+=1