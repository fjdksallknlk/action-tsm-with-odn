import scipy as sp
import os
import pickle as cpk
from scipy.io import loadmat, savemat
from tools.openmax_utils import getlabellist
import json
import argparse
import shutil


def eval_openmax(labeldict, openmax_path, savefile, numclass, sigma, rho):

    correct_open = 0
    tr, fr, ta, fa = 0, 0, 0, 0

    filelist = os.listdir(openmax_path)

    all_samples = len(filelist)
    open_clf_known_samples = 0

    reject_dict = []
    for fname in filelist:
        video_name = fname.split("_")[1]
        open_soft_value = loadmat(openmax_path + fname)
        predict_open = open_soft_value['openmax'].argmax() + 1
        # 0 ==> unknown
        ground_truth = 0
        if video_name in labeldict.keys():
            ground_truth = int(labeldict[video_name])

        if predict_open == numclass + 1:
            # reject
            if video_name in labeldict.keys():
                fr += 1
            else:
                tr += 1
        else:
            if open_soft_value['openmax'].max() >= rho:
                # accept
                open_clf_known_samples += 1
                if video_name not in labeldict.keys():
                    fa += 1
                else:
                    ta += 1
                    if predict_open == ground_truth:
                        # true accept & right predict
                        correct_open += 1
            else:
                # reject
                if video_name in labeldict.keys():
                    fr += 1
                else:
                    tr += 1

    acc = 1.0*correct_open/open_clf_known_samples
    print ('Openmax recognition accuracy: %.4f' % (acc))

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
    if abs(rho - 0.5) < 0.001:
        with open(savefile + '_std', "a") as fh:
            fh.write("FPR: %.4f TPR: %.4f K_P: %.4f K_R: %.4f UK_P: %.4f UK_R: %.4f K_Fscore: %.4f UK_Fscore: %.4f acc: %.4f\n" % (FPR, TPR, k_precision, k_recall, uk_precision, uk_recall, k_fscore, uk_fscore, acc))



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-op', '--openmax_path', type=str,
            default='./output/openmax/tmp/', help='dir where openmax scores are saved')
    parser.add_argument('-sp', '--save_path', type=str,
            default='./output/result/tmp/', help='dir where result are saved')
    parser.add_argument('-cl', '--classlabel', type=str,
            default='./static/classlabel.txt', help='known categories names')
    parser.add_argument('-nc', '--numclass', type=int,
            default=50, help='the known categories num')
    parser.add_argument('-sg', '--sigma', type=int,
            default=14, help='sigma')
    parser.add_argument('--rho', type=float,
            default=0.5, help='rho')

    args = parser.parse_args()

    classlabel_file = args.classlabel
    sigma = args.sigma
    openmax_path = args.openmax_path + '_%d/'%sigma
    savepath = args.save_path
    numclass = args.numclass
    rho = args.rho

    if os.path.isdir(savepath):
        print ("==== Delete old res in dir: %s ====" % savepath)
        shutil.rmtree(savepath)
    os.makedirs(savepath)
    savefile = os.path.join(savepath, 'res.txt')
    

    labeldict = getlabellist(classlabel_file)
    eval_openmax(labeldict, openmax_path, savefile, numclass, sigma, rho)


if __name__ == '__main__':
    main()
