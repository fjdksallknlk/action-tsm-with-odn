import scipy as sp
import glob
import os
import numpy as np
from scipy.io import loadmat, savemat
import argparse
import shutil


def getlabellist(fname):

    videolabels = open(fname, 'r').readlines()
    labeldict = {p.strip().split(' ')[0]: p.strip().split(' ')[1] for p in videolabels}
    return labeldict


def compute_thres(labeldict, feat_path, savefile):

    thres = {}
    for category in labeldict.keys():
        print ('----- Compute the thresholds for category %s -----'%category)
        feat_filelist = glob.glob('%s/v_%s_*' % (feat_path, category))
        correct_features = []
        for feat in feat_filelist:
            try:
                vid_arr = np.loadtxt(feat)
                predicted_category = vid_arr.argmax() + 1
                if predicted_category == int(labeldict[category]):
                    correct_features += [vid_arr]
            except TypeError:
                continue

        correct_features = sp.asarray(correct_features)
        mav = np.mean(correct_features, axis = 0).ravel()
        f, s = mav[np.argsort(-mav)][:2]
        thres[category] = [f, f - s]

    savemat(savefile, thres)


def compute_thres_with_times(labeldict, feat_path, savefile):

    thres = {}
    for category in labeldict.keys():
        print ('----- Compute the thresholds for category %s -----'%category)
        feat_filelist = glob.glob('%s/v_%s_*' % (feat_path, category))
        correct_features = []
        min_times = 10000
        for feat in feat_filelist:
            try:
                vid_arr = loadmat(feat)
                predicted_category = vid_arr['score'].argmax() + 1
                if predicted_category == int(labeldict[category]):
                    correct_features += [vid_arr['score']]
                    score = vid_arr['score'].ravel()
                    f, s = score[np.argsort(-score)][:2]
                    times = f / s
                    # times = vid_arr['dis'].ravel().mean()
                    min_times = times if times < min_times else min_times
            except TypeError:
                continue

        correct_features = sp.asarray(correct_features)
        mav = np.mean(correct_features, axis = 0).ravel()
        f, s = mav[np.argsort(-mav)][:2]
        thres[category] = [f, int(min_times)]

    savemat(savefile, thres)


def compute_thres_times(labeldict, feat_path, savefile):

    thres = {}
    for category in labeldict.keys():
        print ('----- Compute the thresholds for category %s -----'%category)
        feat_filelist = glob.glob('%s/v_%s_*' % (feat_path, category))
        min_times = 10000
        for feat in feat_filelist:
            try:
                vid_arr = loadmat(feat)
                predicted_category = vid_arr['score'].argmax() + 1
                if predicted_category == int(labeldict[category]):
                    times = vid_arr['dis'].ravel().mean()
                    min_times = times if times < min_times else min_times
            except TypeError:
                continue

        thres[category] = int(min_times)

    savemat(savefile, thres)


def compute_thres_stat(labeldict, feat_path, savefile):

    thres = {}
    for category in labeldict.keys():
        print ('----- Compute the thresholds for category %s -----'%category)
        feat_filelist = glob.glob('%s/v_%s_*' % (feat_path, category))
        min_times = []
        for feat in feat_filelist:
            try:
                vid_arr = np.loadtxt(feat)
                predicted_category = vid_arr.argmax() + 1
                score = vid_arr.ravel()
                f, s = score[np.argsort(-score)][:2]
                times = f / s
                min_times.append(times)
            except TypeError:
                continue

        thres[category] = np.asarray(min_times).mean()

    savemat(savefile, thres)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', '--feature_path', type=str,
            default='./features/tmp/', help='dir where the train features are saved')
    parser.add_argument('-sp', '--save_path', type=str,
            default='./output/thres/tmp', help='file where to save the thresholds')
    parser.add_argument('-cl', '--classlabel', type=str,
            default='./static/classlabel.txt', help='class label file')
    parser.add_argument('-tt', '--thres_type', type=str,
            default='./triplet', help='thresholds type: triplet, multiple, triplet_multiple')

    args = parser.parse_args()

    feat_path = args.feature_path
    save_path = args.save_path
    classlabel_file = args.classlabel
    thres_type = args.thres_type

    func_map = {
            'triplet': compute_thres,
            'multiple': compute_thres_times,
            'triplet_multiple': compute_thres_with_times,
            'stat': compute_thres_stat
            }

    if os.path.isdir(save_path):
        print ("==== Delete old thres in dir: %s ====" % save_path)
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    labeldict = getlabellist(classlabel_file)

    compute_func = func_map[thres_type]
    compute_func(labeldict, feat_path, os.path.join(save_path, 'thres.mat'))


if __name__ == '__main__':
    main()
