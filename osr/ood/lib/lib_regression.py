# several functions are from https://github.com/xingjunm/lid_adversarial_subspace_detection
from __future__ import print_function
import numpy as np
import os
from utils import callog

from scipy.spatial.distance import pdist, cdist, squareform

data_split_dict = {
    'ucf11': 490,
    'ucf50': 1116,
    'ucf101': 3783,
    'hmdb51': 1530,
    'ucf11-out': 182,
    'ucf50-out': 536,
    'ucf101-out': 1888,
    'hmdb51-out': 780,
}


def block_split(X, Y, inData, outData):
    """
    Split the data training and testing
    :return: X (data) and Y (label) for training / testing
    """
    num_samples = X.shape[0]

    partition = data_split_dict[outData]

    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: :], Y[partition: :]
    num_train = 100

    X_train = np.concatenate((X_norm[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test


def detection_performance(regressor, X, Y, outf):
    """
    Measure the detection performance
    return: detection metrics
    """
    num_samples = X.shape[0]
    l1 = open('%s/confidence_TMP_In.txt'%outf, 'w')
    l2 = open('%s/confidence_TMP_Out.txt'%outf, 'w')
    y_pred = regressor.predict_proba(X)[:, 1]

    for i in range(num_samples):
        if Y[i] == 0:
            l1.write("{}\n".format(-y_pred[i]))
        else:
            l2.write("{}\n".format(-y_pred[i]))
    l1.close()
    l2.close()
    results = callog.metric(outf, ['TMP'])
    return results


def load_characteristics(score, dataset, out, outf):
    """
    Load the calculated scores
    return: data and label of input score
    """
    X, Y = None, None

    file_name = os.path.join(outf, "%s_%s_%s.npy" % (score, dataset, out))
    data = np.load(file_name)

    if X is None:
        X = data[:, :-1]
    else:
        X = np.concatenate((X, data[:, :-1]), axis=1)
    if Y is None:
        Y = data[:, -1] # labels only need to load once

    return X, Y
