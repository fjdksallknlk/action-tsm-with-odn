import scipy as sp
import glob
import os
import numpy as np
from scipy.io import loadmat, savemat
import argparse
import shutil


def getlabellist(fname):

    videolabels = open(fname, 'r').readlines()
    labeldict = {p.strip().split(" ")[0]: p.strip().split(" ")[1] for p in videolabels}
    return labeldict


def compute_mean_vector(category_name, labeldict, featpath, savepath, layer = 'fc'):

    channel_nums = 1

    featurefile_list = glob.glob('%s/v_%s_*' % (featpath, category_name))
    correct_features = []
    for featurefile in featurefile_list:
        try:
            # vid_arr = loadmat(featurefile)
            vid_arr = np.loadtxt(featurefile)
            predicted_category = vid_arr.argmax()

            if predicted_category == int(labeldict[category_name]) - 1:
                correct_features += [vid_arr]

        except TypeError:
            continue

    channel_mean_vec = []
    for channelid in range(channel_nums):
        channel = []
        for feature in correct_features:
            channel += [feature]
        channel = sp.asarray(channel)
        assert len(correct_features) == channel.shape[0]
        channel_mean_vec += [sp.mean(channel, axis=0)]

    channel_mean_vec = sp.asarray(channel_mean_vec)
    savemat(savepath + '/%s.mat' % category_name, {'%s' % category_name: channel_mean_vec})


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', '--feature_path', type=str,
            default='./features/tmp', help='feature path.')
    parser.add_argument('-sp', '--save_path', type=str,
            default='./output/mavs/tmp', help='save path.')
    parser.add_argument('-cl', '--classlabel', type=str,
            default='./static/classlabel.txt', help='class label file.')
    args = parser.parse_args()

    featpath = args.feature_path
    classlabel_file = args.classlabel
    savepath = args.save_path
    if os.path.isdir(savepath):
        print ("==== Delete old mavs in dir: %s ====" % savepath)
        shutil.rmtree(savepath)
    os.makedirs(savepath)

    labeldict = getlabellist(classlabel_file)
    for category_name in labeldict.keys():
        print ("Compute the mav for category %s." % category_name)
        compute_mean_vector(category_name, labeldict, featpath, savepath)


if __name__ == '__main__':
    main()
