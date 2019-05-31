import time

import record
import model

RUNTIME_SAMPLE_FILENAME = 'temp.wav'

def make_clsfr():
    print("Processing samples...")
    feat, label = model.create_training_info('samples_train.csv')
    print("Learning...")
    clsfr = model.create_clsfr()
    clsfr = model.train_clsfr(clsfr, feat, label)
    print("Saving...")
    model.write_clsfr(clsfr)
    print("Saved!")

def perform_test():
    print("Recording starts in 3 sec!")
    time.sleep(3)
    frames = record.record_frames()
    record.write_frames(frames, RUNTIME_SAMPLE_FILENAME)
    print("Loading classifier...")
    clsfr = model.read_clsfr()
    print("Running test...")
    verdict, score = model.determine_from_sample(RUNTIME_SAMPLE_FILENAME, clsfr)
    print("TEST {}".format("SUCCESSFUL" if verdict else "FAILED"))
    print("Score was {}".format(score))
