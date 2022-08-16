import scipy as sp
import glob
import os
import numpy as np
import scipy.spatial.distance as spd
from scipy.io import loadmat, savemat
import argparse
import shutil


Channel_Nums = 1

def getlabellist(fname):

    videolabels = open(fname, 'r').readlines()
    labeldict = {p.strip().split(" ")[0]: p.strip().split(" ")[1] for p in videolabels}
    return labeldict


def compute_channel_distances(mean_feature_vec, features, category_name):

    eucos_dist, eu_dist, cos_dist = [], [], []
    for channel in range(Channel_Nums):
        eu_channel, cos_channel, eu_cos_channel = [], [], []
        for feat in features:
            eu_channel += [spd.euclidean(mean_feature_vec[category_name][channel, :], feat)]
            cos_channel += [spd.cosine(mean_feature_vec[category_name][channel, :], feat)]
            eu_cos_channel += [spd.euclidean(mean_feature_vec[category_name][channel, :], feat)/200. +
                               spd.cosine(mean_feature_vec[category_name][channel, :], feat)]
        eu_dist += [eu_channel]
        cos_dist += [cos_channel]
        eucos_dist += [eu_cos_channel]

    eu_dist = sp.asarray(eu_dist)
    cos_dist = sp.asarray(cos_dist)
    eucos_dist = sp.asarray(eucos_dist)

    assert eucos_dist.shape[0] == Channel_Nums
    assert eu_dist.shape[0] == Channel_Nums
    assert cos_dist.shape[0] == Channel_Nums
    assert eucos_dist.shape[1] == len(features)
    assert eu_dist.shape[1] == len(features)
    assert cos_dist.shape[1] == len(features)

    channel_distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean': eu_dist}
    return channel_distances


def compute_distances(category_name, labeldict, mav_path, feat_path, save_path, layer = 'fc'):

    mean_feature_vec = loadmat(mav_path + '/%s.mat' % category_name)
    featurefile_list = glob.glob('%s/v_%s_*' % (feat_path, category_name))

    correct_features = []
    for featurefile in featurefile_list:
        try:
            vid_arr = np.loadtxt(featurefile)
            predicted_category = vid_arr.argmax() + 1
            if predicted_category == int(labeldict[category_name]):
                correct_features += [vid_arr]
        except TypeError:
            continue

    distance_distribution = compute_channel_distances(mean_feature_vec, correct_features, category_name)
    savemat(save_path + '/%s_dis.mat' % category_name, distance_distribution)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--mav_path', type=str,
            default='./output/mavs/tmp/', help='dir where mavs are saved')
    parser.add_argument('-fp', '--feature_path', type=str,
            default='./features/tmp/', help='dir where feats are saved')
    parser.add_argument('-sp', '--save_path', type=str,
            default='./output/dis/tmp/', help='dir where to save the distances')
    parser.add_argument('-cl', '--classlabel', type=str,
            default='./static/classlabel.txt')

    args = parser.parse_args()

    classlabel_file = args.classlabel
    mav_path = args.mav_path
    feat_path = args.feature_path
    save_path = args.save_path

    if os.path.isdir(save_path):
        print ("==== Delete old dis in dir: %s ====" % save_path)
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    labeldict = getlabellist(classlabel_file)
    for category_name in labeldict.keys():
        print ("Compute the dis for category %s." % category_name)
        compute_distances(category_name, labeldict, mav_path, feat_path, save_path)


if __name__ == '__main__':
    main()
