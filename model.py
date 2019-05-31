from csv import DictReader
from contextlib import closing
import numpy as np
import pickle
import analyze

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def create_clsfr(verbose=False):
    return MLPClassifier(hidden_layer_sizes=(50,25), verbose=verbose, max_iter=2000)

def train_clsfr(clsfr, x, y):
    clsfr.fit(x,y)
    return clsfr

def score_clsfr(clsfr, x, y):
    y_pred = clsfr.predict(x)
    score = accuracy_score(y, y_pred)
    return score

def write_clsfr(clsfr):
    with closing(open('clsfr.pkl','wb')) as pklfile:
        pickle.dump(clsfr, pklfile)

def read_clsfr():
    with closing(open('clsfr.pkl','rb')) as pklfile:
        clsfr= pickle.load(pklfile)
    return clsfr

def create_training_info(filename):
    feat_finale = None
    labels_finale = None
    with closing(open(filename,'r')) as csvfile:
        dr = DictReader(csvfile)
        for row in dr:
            freq, wf = analyze.open_wave(row['filename'])
            feat = analyze.extract_features(freq, wf)
            labels = np.ones((feat.shape[0],1)) * float(row['is_ok'])
            if feat_finale is None:
                feat_finale = feat
            else:
                feat_finale = np.vstack((feat_finale,feat))
            if labels_finale is None:
                labels_finale = labels
            else:
                labels_finale = np.vstack((labels_finale, labels))
    return feat_finale, labels_finale