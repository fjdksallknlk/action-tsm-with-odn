from multiprocessing import Pool
import os, time, random
from eval_thres import eval_triplet
from tools.openmax_utils import getlabellist
import numpy as np
import argparse
from scipy.io import loadmat
import shutil


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--save_path', type=str,
            default='./output/result/thres/tmp', help='file to save thres evaluation results')
    parser.add_argument('-fp', '--feature_path', type=str,
            default='./features/tmp/', help='dir where to save features')
    parser.add_argument('-cl', '--classlabel', type=str,
            default='./static/classlabel.txt', help='class label file')
    parser.add_argument('-tf', '--thres_file', type=str,
            default='./output/thres/tmp.mat', help='thresholds file')
    parser.add_argument('-tt', '--thres_type', type=str,
            default='triplet', help='eval method for diff thres type')

    args = parser.parse_args()

    feature_path = args.feature_path
    classlabel_file = args.classlabel
    save_path = args.save_path
    thres_file = args.thres_file
    thres_type = args.thres_type

    labeldict = getlabellist(classlabel_file)
    thres = loadmat(thres_file)

    if os.path.isdir(save_path):
        print ("==== Delete old res in dir: %s ====" % save_path)
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    savefile = os.path.join(save_path, 'res.txt')

    print ('Parent process %s.' % os.getpid())
    p = Pool()

    cmd_offset = 14
    for alpha in np.arange(0.1, 1.2, 0.1):
        for beta in np.arange(0.0, alpha + 0.1, 0.1):
            for sigma in np.arange(0.1, 1.2, 0.1):
                p.apply_async(
                    eval_triplet,
                    args=(feature_path, savefile, labeldict, thres, alpha, beta, sigma)
                )

    print ('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print ('All subprocesses done.')
    cmd = 'sort %s -n -k %d > ./tmp' % (savefile, cmd_offset)
    os.system(cmd)
    cmd = 'mv ./tmp %s' % savefile
    os.system(cmd)


if __name__=='__main__':

    main()
