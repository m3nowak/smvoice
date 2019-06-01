import os.path
import pickle
from contextlib import closing
from csv import DictReader

import numpy as np
from sklearn.mixture import GaussianMixture

import analyze

ARTIFACT_DIR = 'artifacts/gmm'
TRAINING_INFO_FILE = 'training.csv'
TEST_INFO_FILE = 'test.csv'
GM_MODEL_FILE = 'gmm.pkl'

class TestResult:
    def __init__(self, filename, score, is_ok):
        self.filename = filename
        self.score = score
        self.is_ok = is_ok
    
    def __str__(self):
        is_ok_cap = "Good" if self.is_ok else "Bad"
        return "{} sample named {} result {}".format(is_ok_cap, self.filename, self.score)

def generate_features():
    full_csv_path = os.path.join(ARTIFACT_DIR, TRAINING_INFO_FILE)
    feat_stacked = None
    with closing(open(full_csv_path, 'r')) as csvfile:
        dr = DictReader(csvfile)
        for row in dr:
            sample_path = os.path.join(ARTIFACT_DIR, row['filename'])
            feat = analyze.extract_features_file(sample_path)
            if feat_stacked is None:
                feat_stacked = feat
            else:
                feat_stacked = np.vstack((feat_stacked, feat))
    return feat_stacked

def train_model(verbose=2):
    gm = GaussianMixture(n_components=32, max_iter = 200, n_init=3, covariance_type='diag', verbose=verbose, tol=5e-4)
    feat = generate_features()
    gm.fit(feat)
    return gm

def write_model(gm):
    filename = os.path.join(ARTIFACT_DIR, GM_MODEL_FILE)
    with closing(open(filename, 'wb')) as pkl_file:
        pickle.dump(gm, pkl_file)

def read_model():
    filename = os.path.join(ARTIFACT_DIR, GM_MODEL_FILE)
    with closing(open(filename, 'rb')) as pkl_file:
        gm = pickle.load(pkl_file)
    return gm

def sample_test(gm, sample_filename):
    feat = analyze.extract_features_file(sample_filename)
    return gm.score(feat)

def test_model(gm):
    full_csv_path = os.path.join(ARTIFACT_DIR, TEST_INFO_FILE)
    result = []
    with closing(open(full_csv_path, 'r')) as csvfile:
        dr = DictReader(csvfile)
        for row in dr:
            full_sample_path = os.path.join(ARTIFACT_DIR, row['filename'])
            feat = analyze.extract_features_file(full_sample_path)
            score = gm.score(feat)
            is_ok = row['is_ok'] == '1'
            result.append(TestResult(row['filename'],score,is_ok))
    return result