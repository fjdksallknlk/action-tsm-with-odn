# -*- coding: utf-8 -*-

import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
import numpy as np
from scipy.io import loadmat, savemat

from tools.openmax_utils import *
from tools.evt_fitting import weibull_tailfitting, query_weibull

import shutil

try:
    import libmr
except ImportError:
    print ("LibMR not installed or libmr.so not found")
    print ("Install libmr: cd libMR/; ./compile.sh")
    sys.exit()


#---------------------------------------------------------------------------------
# params and configuratoins
NCHANNELS = 1
ALPHA_RANK = 1
WEIBULL_TAIL_SIZE = 20
SIGMA = 14

#---------------------------------------------------------------------------------
def computeOpenMaxProbability(NCLASSES, openmax_fc, openmax_score_u):

    prob_scores, prob_unknowns = [], []
    for channel in range(NCHANNELS):
        channel_scores, channel_unknowns = [], []
        for category in range(NCLASSES):
            channel_scores += [sp.exp(openmax_fc[channel, category])]

        # total_denominator = sp.sum(sp.exp(openmax_fc[channel, :])) + sp.exp(sp.sum(openmax_score_u[channel, :]))
        total_denominator = sp.sum(sp.exp(openmax_fc[channel, :])) + sp.sum(sp.exp(openmax_score_u[channel, :]))
        prob_scores += [channel_scores/total_denominator ]
        # prob_unknowns += [sp.exp(sp.sum(openmax_score_u[channel, :]))/total_denominator]
        prob_unknowns += [sp.sum(sp.exp(openmax_score_u[channel, :]))/total_denominator]

    prob_scores = sp.asarray(prob_scores)
    prob_unknowns = sp.asarray(prob_unknowns)

    scores = sp.mean(prob_scores, axis = 0)
    unknowns = sp.mean(prob_unknowns, axis=0)
    modified_scores =  scores.tolist() + [unknowns]
    assert len(modified_scores) == NCLASSES + 1

    return modified_scores

#---------------------------------------------------------------------------------
def recalibrate_scores(NCLASSES, weibull_model, labeldict, video_arr, sigma,
                       layer = 'fc', alpharank = 1, distance_type = 'eucos'):

    video_layer = video_arr[layer]
    ranked_list = video_arr['score'].argsort().ravel()[::-1]
    alpha_weights = [((alpharank+1) - i)/float(sigma*alpharank) for i in range(1, alpharank+1)]
    ranked_alpha = sp.zeros(101)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Now recalibrate each fc score for each channel and for each class
    # to include probability of unknown
    # openmax_fc, openmax_score_u = [], []
    openmax_fc, openmax_score_u, wscore_fc = [], [], []
    video_score = video_arr['score']
    predict_label = video_score.argmax()
    for channel in range(NCHANNELS):
        channel_scores = video_layer[channel, :]
        openmax_fc_channel = []
        openmax_fc_unknown = []
        wscore_channel = []
        count = 0
        for categoryid in range(NCLASSES):
            category_name = list(labeldict.keys())[list(labeldict.values()).index(str(categoryid + 1))]
            # category_name = labeldict.keys()[labeldict.values().index(str(categoryid + 1))]
            # get distance between current channel and mean vector
            category_weibull = query_weibull(category_name, weibull_model, distance_type = distance_type)
            channel_distance = compute_distance(channel_scores, channel, category_weibull[0],
                                                distance_type = distance_type)
            # obtain w_score for the distance and compute probability of the distance
            # being unknown wrt to mean training vector and channel distances for
            # category and channel under consideration
            wscore = category_weibull[2][channel].w_score(channel_distance)

            if categoryid == predict_label:
                wscore_channel += [wscore]

            modified_fc_score = channel_scores[categoryid] * ( 1 - wscore*ranked_alpha[categoryid] )
            openmax_fc_channel += [modified_fc_score]
            openmax_fc_unknown += [channel_scores[categoryid] - modified_fc_score ]
        # gather modified scores fc scores for each channel for the given image
        openmax_fc += [openmax_fc_channel]
        openmax_score_u += [openmax_fc_unknown]
        wscore_fc += [wscore_channel]

    openmax_fc = sp.asarray(openmax_fc)
    openmax_score_u = sp.asarray(openmax_score_u)
    wscore_fc = sp.asarray(wscore_fc)

    # Pass the recalibrated fc scores for the image into openmax
    openmax_probab = computeOpenMaxProbability(NCLASSES, openmax_fc, openmax_score_u)
    softmax_probab = video_arr['score'].ravel()
    wscore_fc_mean = wscore_fc.ravel().mean()
    return sp.asarray(openmax_probab), sp.asarray(softmax_probab), wscore_fc_mean

#---------------------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()


    # Optional arguments.
    parser.add_argument(
        "--weibull_tailsize",
        type=int,
        default=WEIBULL_TAIL_SIZE,
        help="Tail size used for weibull fitting"
    )

    parser.add_argument(
        "--alpha_rank",
        type=int,
        default=ALPHA_RANK,
        help="Alpha rank to be used as a weight multiplier for top K scores"
    )

    parser.add_argument(
        "--sigma",
        type=int,
        default=SIGMA,
        help="Weight of alpha rank"
    )

    parser.add_argument(
        "--distance",
        default='eucos',
        help="Type of distance to be used for calculating distance \
        between mean vector and query image \
        (eucos, cosine, euclidean)"
    )

    parser.add_argument(
        "--mean_files_path",
        default='./data/category_mav/',
        help="Path to directory where mean activation vector (MAV) is saved."
    )

    parser.add_argument(
        "--category_name",
        default='./classlabel.txt',
        help="Path to category name"
    )

    parser.add_argument(
        "--video_feat_path",
        default='./data/test_features/',
        help="Video Array name for which openmax scores are to be computed"
    )

    parser.add_argument(
        "--distance_path",
        default='./data/category_dis/',
        help="Path to directory where distances of training data \
        from Mean Activation Vector is saved"
    )

    parser.add_argument(
        "--save_path",
        default='./data/openmax',
        help="Path to directory where openmax scores are saved"
    )

    parser.add_argument(
        "--num_classes", type=int,
        default=101,
        help="Known class num"
    )

    args = parser.parse_args()

    distance_path = args.distance_path
    mean_path = args.mean_files_path
    video_feat_path = args.video_feat_path
    category_name = args.category_name
    save_path = args.save_path
    NCLASSES = args.num_classes

    alpha_rank = args.alpha_rank
    weibull_tailsize = args.weibull_tailsize
    sigma = args.sigma

    save_path = save_path + '_%d'%sigma + '/'

    if os.path.isdir(save_path):
        print ("==== Delete old openmax scores in dir %s ====" % save_path)
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    labeldict = getlabellist(category_name)
    weibull_model = weibull_tailfitting(mean_path, distance_path, labeldict,
                                        tailsize = WEIBULL_TAIL_SIZE)

    print ("Completed Weibull fitting on %s models" %len(weibull_model.keys()))

    for featurefile in os.listdir(video_feat_path):
        video_arr = np.loadtxt(os.path.join(video_feat_path + featurefile))

        video_arr = {'fc': sp.asarray([video_arr]), 'score': video_arr}
        
        openmax, softmax, wscore =  recalibrate_scores(NCLASSES, weibull_model, labeldict, video_arr, sigma)
        print ("Video ArrName: %s" % featurefile)
        savemat(save_path + '%s' % featurefile, {'softmax': softmax, 'openmax': openmax, 'wscore': wscore})
        # print ("Softmax Scores ", softmax)
        # print ("Openmax Scores ", openmax)
        # print (openmax.shape, softmax.shape)


if __name__ == "__main__":
    main()
