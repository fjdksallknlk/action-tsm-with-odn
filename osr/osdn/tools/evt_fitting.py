# -*- coding: utf-8 -*-

import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
from scipy.io import loadmat

from tools.openmax_utils import *

try:
    import libmr
except ImportError:
    print ("LibMR not installed or libmr.so not found here")
    print ("Install libmr: cd libMR/; ./compile.sh")
    sys.exit()

#---------------------------------------------------------------------------------
NCHANNELS = 1

#---------------------------------------------------------------------------------
def weibull_tailfitting(meanfiles_path, distancefiles_path, labellist,
                        tailsize = 20,
                        distance_type = 'eucos'):

    weibull_model = {}
    # for each category, read meanfile, distance file, and perform weibull fitting
    for category in labellist.keys():
        weibull_model[category] = {}
        distance_scores = loadmat('%s/%s_dis.mat' %(distancefiles_path, category))[distance_type]
        meantrain_vec = loadmat('%s/%s.mat' %(meanfiles_path, category))

        weibull_model[category]['distances_%s'%distance_type] = distance_scores
        weibull_model[category]['mean_vec'] = meantrain_vec
        weibull_model[category]['weibull_model'] = []
        for channel in range(NCHANNELS):
            mr = libmr.MR()
            tailtofit = sorted(distance_scores[channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category]['weibull_model'] += [mr]

    return weibull_model

#---------------------------------------------------------------------------------
def query_weibull(category_name, weibull_model, distance_type = 'eucos'):

    category_weibull = []
    category_weibull += [weibull_model[category_name]['mean_vec'][category_name]]
    category_weibull += [weibull_model[category_name]['distances_%s' %distance_type]]
    category_weibull += [weibull_model[category_name]['weibull_model']]

    return category_weibull

