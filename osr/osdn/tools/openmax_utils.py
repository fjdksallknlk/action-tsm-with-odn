# -*- coding: utf-8 -*-

import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp


def parse_synsetfile(synsetfname):
    """ Read ImageNet 2012 file
    """
    categorylist = open(synsetfname, 'r').readlines()
    imageNetIDs = {}
    count = 0
    for categoryinfo in categorylist:
        wnetid = categoryinfo.split(' ')[0]
        categoryname = ' '.join(categoryinfo.split(' ')[1:])
        imageNetIDs[str(count)] = [wnetid, categoryname]
        count += 1

    assert len(imageNetIDs.keys()) == 1000
    return imageNetIDs

def getlabellist(fname):

    categorylist = open(fname, 'r').readlines()
    labeldict = {p.strip().split(' ')[0]: p.strip().split(' ')[1] for p in categorylist}
    return labeldict


def compute_distance(query_channel, channel, mean_vec, distance_type = 'eucos'):

    if distance_type == 'eucos':
        query_distance = spd.euclidean(mean_vec[channel, :], query_channel)/200. + spd.cosine(mean_vec[channel, :], query_channel)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mean_vec[channel, :], query_channel)/200.
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mean_vec[channel, :], query_channel)
    else:
        print ("distance type [%s] not known: enter either of eucos, euclidean or cosine"%distance_type)
    return query_distance
