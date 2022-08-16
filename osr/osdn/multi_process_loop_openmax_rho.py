from multiprocessing import Pool
import scipy as sp
import os
import pickle as cpk
from scipy.io import loadmat, savemat
from tools.openmax_utils import getlabellist
import json
import argparse
import shutil
import numpy as np

from eval_openmax import eval_openmax


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

    if os.path.isdir(savepath):
        print ("==== Delete old res in dir: %s ====" % savepath)
        shutil.rmtree(savepath)
    os.makedirs(savepath)
    savefile = os.path.join(savepath, 'res.txt')

    labeldict = getlabellist(classlabel_file)

    print ('Parent process %s.' % os.getpid())
    p = Pool()

    for rho in np.arange(0.0, 1.0, 0.001):
        p.apply_async(
            eval_openmax,
            args=(labeldict, openmax_path, savefile, numclass, sigma, rho)
        )

    print ('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print ('All subprocesses done.')

    cmd_offset = 14
    cmd = 'sort %s -n -k %d > ./tmp' % (savefile, cmd_offset)
    os.system(cmd)
    cmd = 'mv ./tmp %s' % savefile
    os.system(cmd)


if __name__ == '__main__':
    main()
