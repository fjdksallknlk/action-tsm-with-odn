# -*- coding: utf8 -*-
import os


def str2bool(v):
    if v.lower() in ['y', '1', 'yes', 'true', 't']:
        return True
    return False

def str2list(v):
    return [int(val) for val in v.split(',')]

def may_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
