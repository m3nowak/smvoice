from os import name, system
import time

import numpy as np
from scipy.io.wavfile import read as read_wav

import analyze
import record
import model
import user_commands
import audio_conv

import gmm_model

import gmm_user_commands as gmuc

def clear(): 
  
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 

def main():
    choice = None
    prompt = """1: uruchom test
2: naucz model
3: wykonaj test modelu na zgromadzonych próbkach
4: ustaw próg rozpoznawania
5: wyjdź"""
    while choice != "5":
        clear()
        print(prompt)
        choice = input(">")
        if choice == "1":
            gmuc.interactive_test()
        elif choice == "2":
            gmuc.create_model()
        elif choice == "3":
            gmuc.test_run()
        elif choice == "4":
            gmuc.write_threshold()
        elif choice != "5":
            input("Zły wybór, wciśnij ENT")
        if choice != "5":
            input("ENT aby kontunuować")

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
