import time

import record
import model
from contextlib import closing
from csv import DictReader

RUNTIME_SAMPLE_FILENAME = 'temp.wav'

def make_clsfr():
    print("Processing samples...")
    feat, label = model.create_features_data('samples_train.csv')
    print("Learning...")
    clsfr = model.create_clsfr()
    clsfr = model.train_clsfr(clsfr, feat, label)
    print("Saving...")
    model.write_clsfr(clsfr)
    print("Saved!")

def perform_live_test():
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

def test_trial():
    print("Loading classifier...")
    clsfr = model.read_clsfr()
    print("Running trial...")
    with closing(open('samples_test.csv','r')) as csvfile:
        dr = DictReader(csvfile)
        for row in dr:
            print("Testing {} ...".format(row['filename']))
            print("Expecting {} result.".format("possitive" if row['is_ok'] == '1' else "negative"))
            verdict, score = model.determine_from_sample(row['filename'], clsfr)
            print("{} result. Score is {}".format("possitive" if verdict else "negative", score))
            print("="*10)