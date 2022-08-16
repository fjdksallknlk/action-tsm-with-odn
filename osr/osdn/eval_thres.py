import scipy as sp
import os
import sys
import numpy as np
import pickle as cpk
from scipy.io import loadmat, savemat
from tools.openmax_utils import getlabellist


def eval_triplet(feature_path, savefile, labeldict, thres, alpha, beta, sigma):

    print ("alpha: %f, beta: %f, sigma: %f"%(alpha, beta, sigma))
    correct_feat = 0
    triplet_clf_known_samples = 0

    tr, fr, ta, fa = 0, 0, 0, 0

    filelist_ = os.listdir(feature_path)
    filelist = sorted(filelist_)

    # for fname in os.listdir(path):
    for fname in filelist:
        video_name = fname.split('_')[1]
        feat = np.loadtxt(feature_path + fname).ravel()
        ground_truth = int(labeldict[video_name]) if video_name in labeldict.keys() else 0
        ground_truth_name = video_name
        predict = feat.argmax() + 1
        predict_name = list(labeldict.keys())[list(labeldict.values()).index(str(predict))]
        thres_category = thres[predict_name].ravel()
        f, s = feat[np.argsort(-feat)][:2]

        if f > thres_category[0] * alpha:
            # accept
            triplet_clf_known_samples += 1
            if predict == ground_truth:
                correct_feat += 1
            if ground_truth_name in labeldict.keys():
                # true accept
                ta += 1
            else:
                # false accept
                fa += 1
        elif f < thres_category[0] * beta:
            # reject
            if ground_truth_name in labeldict.keys():
                # false reject
                fr += 1
            else:
                # true reject
                tr += 1
        else:
            # judge distance
            dis = f - s
            if dis > thres_category[1] * sigma:
                # accept
                triplet_clf_known_samples += 1
                if predict == ground_truth:
                    correct_feat += 1
                if ground_truth_name in labeldict.keys():
                    # true accept
                    ta += 1
                else:
                    # false accept
                    fa += 1
            else:
                # reject or judge by openmax
                if ground_truth_name in labeldict.keys():
                    # false reject
                    fr += 1
                else:
                    # true reject
                    tr += 1

    acc = 1.0*correct_feat/triplet_clf_known_samples
    print ('Triplet recognition accuracy: %.4f' % (acc))

    # known is positive
    TP, FP, FN, TN = ta, fa, fr, tr

    FPR = FP*1.0 / (FP + TN)  # fa / (fa + tr)
    TPR = TP*1.0 / (TP + FN)  # ta / (ta + fr)

    k_precision = TP*1.0 / (TP + FP)
    k_recall = TP*1.0 / (TP + FN)
    k_fscore = k_precision * k_recall * 2.0 / (k_precision + k_recall)
    print ('Precision: %.4f, Recall: %.4f, Fscore: %.4f' % (k_precision, k_recall, k_fscore))

    # unknown is positive
    TP, FP, FN, TN = tr, fr, fa, ta

    uk_precision = TP*1.0 / (TP + FP)
    uk_recall = TP*1.0 / (TP + FN)
    uk_fscore = uk_precision * uk_recall * 2.0 / (uk_precision + uk_recall)

    with open(savefile, "a") as fh:
        fh.write("FPR: %.4f TPR: %.4f K_P: %.4f K_R: %.4f UK_P: %.4f UK_R: %.4f K_Fscore: %.4f UK_Fscore: %.4f acc: %.4f\n" % (FPR, TPR, k_precision, k_recall, uk_precision, uk_recall, k_fscore, uk_fscore, acc))


def main():

    alpha, beta, sigma = 0.8, 0.8, 1.1
    savefile = './output/result/tmp'
    classlabel = './static/ucf11_known_classlist'
    path = './features/ucf11_gs_gcpl_testlist_dist_norm/'

    if len(sys.argv) > 1:
        alpha = float(sys.argv[1])
        beta = float(sys.argv[2])
        sigma = float(sys.argv[3])
        print ('%.2f %.2f %.2f' % (alpha, beta, sigma))

    labeldict = getlabellist(classlabel)
    thres = loadmat('./output/thres/ucf11_gs_gcpl_thres_dist_norm.mat')
    eval_triplet(path, savefile, labeldict, thres, alpha, beta, sigma)
 


if __name__ == '__main__':
    main()
