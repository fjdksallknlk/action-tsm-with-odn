# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

ROOT_DATASET = '/home/shuyu/data/'

def return_ucf11(modality):
    filename_categories = 'split_list_for_odn/ucf11/static/classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'ucf11/test_frames_tvl1'
        # filename_imglist_train = 'split_list_for_odn/ucf11/init_train/init_trainlist.6_tsm'
        filename_imglist_train = 'split_list_for_odn/ucf11/static/trainlist_tsm'
        filename_imglist_val = 'split_list_for_odn/ucf11/test/testlist_tsm'
        prefix = 'flow_i_{:04d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'ucf11/test_frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf11/static/trainlist_01'
        filename_imglist_val = 'split_list_for_odn/ucf11/test/testlist_01'
        prefix = 'flow_{:s}_{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ucf11_in(modality):
    filename_categories = 'split_list_for_odn/ucf11/static/knownclass'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'ucf11/test_frames_tvl1'
        # filename_imglist_train = 'split_list_for_odn/ucf11/init_train/init_trainlist_tsm'
        filename_imglist_train = 'split_list_for_odn/ucf11/init_train/init_trainlist.6_tsm'
        filename_imglist_val = 'split_list_for_odn/ucf11/test/init_testlist_tsm'
        prefix = 'flow_i_{:04d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'ucf11/test_frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf11/static/trainlist_01'
        filename_imglist_val = 'split_list_for_odn/ucf11/test/testlist_01'
        prefix = 'flow_{:s}_{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_ucf11_out(modality):
    filename_categories = 5
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'ucf11/test_frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf11/init_train/init_trainlist_tsm'
        filename_imglist_val = 'split_list_for_odn/ucf11/test/out_testlist_tsm'
        prefix = 'flow_i_{:04d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'ucf11/test_frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf11/static/trainlist_01'
        filename_imglist_val = 'split_list_for_odn/ucf11/test/testlist_01'
        prefix = 'flow_{:s}_{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ucf50(modality):
    filename_categories = 'split_list_for_odn/ucf50/setting1/static/classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'ucf50/frames_tvl1'
        # filename_imglist_train = 'split_list_for_odn/ucf50/setting1/init_train/init_trainlist.25_tsm'
        filename_imglist_train = 'split_list_for_odn/ucf50/setting1/static/trainlist_tsm'
        filename_imglist_val = 'split_list_for_odn/ucf50/setting1/test/testlist_tsm'
        prefix = 'flow_i_{:04d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'ucf101/frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf50/setting1/static/trainlist_01'
        filename_imglist_val = 'split_list_for_odn/ucf50/setting1/test/testlist_01'
        prefix = 'flow_{:s}_{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ucf50_in(modality):
    filename_categories = 'split_list_for_odn/ucf50/setting1/static/knownclass'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'ucf50/frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf50/setting1/init_train/init_trainlist_tsm'
        # filename_imglist_train = 'split_list_for_odn/ucf50/setting1/init_train/init_trainlist.25_tsm'
        filename_imglist_val = 'split_list_for_odn/ucf50/setting1/test/init_testlist_tsm'
        prefix = 'flow_i_{:04d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'ucf101/frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf50/setting1/static/trainlist_01'
        filename_imglist_val = 'split_list_for_odn/ucf50/setting1/test/testlist_01'
        prefix = 'flow_{:s}_{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ucf50_out(modality):
    filename_categories = 25
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'ucf50/frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf50/setting1/init_train/init_trainlist_tsm'
        filename_imglist_val = 'split_list_for_odn/ucf50/setting1/test/out_testlist_tsm'
        prefix = 'flow_i_{:04d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'ucf101/frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf50/setting1/static/trainlist_01'
        filename_imglist_val = 'split_list_for_odn/ucf50/setting1/test/testlist_01'
        prefix = 'flow_{:s}_{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ucf101(modality):
    filename_categories = 'split_list_for_odn/ucf101/split1/setting1/static/classInd_1'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'ucf101/frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf101/split1/setting1/static/trainlist_01_tsm'
        filename_imglist_val = 'split_list_for_odn/ucf101/split1/setting1/test/testlist_01_tsm'
        prefix = 'flow_i_{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'ucf101/frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf101/split1/setting1/static/trainlist_01'
        filename_imglist_val = 'split_list_for_odn/ucf101/split1/setting1/test/testlist_01'
        prefix = 'flow_{:s}_{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ucf101_in(modality):
    filename_categories = 'split_list_for_odn/ucf101/split1/setting1/static/knownclass'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'ucf101/frames_tvl1'
        # filename_imglist_train = 'split_list_for_odn/ucf101/split1/setting1/init_train/init_trainlist_tsm'
        filename_imglist_train = 'split_list_for_odn/ucf101/split1/setting1/init_train/init_trainlist.50_tsm'
        filename_imglist_val = 'split_list_for_odn/ucf101/split1/setting1/test/init_testlist_tsm'
        prefix = 'flow_i_{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'ucf101/frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf101/split1/setting1/static/trainlist_01'
        filename_imglist_val = 'split_list_for_odn/ucf101/split1/setting1/test/testlist_01'
        prefix = 'flow_{:s}_{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_ucf101_out(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'ucf101/frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf101/split1/setting1/init_train/init_trainlist_tsm'
        filename_imglist_val = 'split_list_for_odn/ucf101/split1/setting1/test/out_testlist_tsm'
        prefix = 'flow_i_{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'ucf101/frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/ucf101/split1/setting1/static/trainlist_01'
        filename_imglist_val = 'split_list_for_odn/ucf101/split1/setting1/test/testlist_01'
        prefix = 'flow_{:s}_{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'Hmdb51/frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/hmdb51/split1/static/trainlist01_tsm'
        filename_imglist_val = 'split_list_for_odn/hmdb51/split1/static/testlist01_tsm'
        prefix = 'flow_i_{:04d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'hmdb/frames_tvl1'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{:s}_{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51_in(modality):
    filename_categories = 25
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'Hmdb51/frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/hmdb51/split1/init_train/init_trainlist_tsm'
        # filename_imglist_train = 'split_list_for_odn/hmdb51/split1/init_train/init_trainlist.25_tsm'
        filename_imglist_val = 'split_list_for_odn/hmdb51/split1/test/init_testlist_tsm'
        prefix = 'flow_i_{:04d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'hmdb/frames_tvl1'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{:s}_{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_hmdb51_out(modality):
    filename_categories = 26
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'Hmdb51/frames_tvl1'
        filename_imglist_train = 'split_list_for_odn/hmdb51/split1/init_train/init_trainlist_tsm'
        filename_imglist_val = 'split_list_for_odn/hmdb51/split1/test/out_testlist_tsm'
        prefix = 'flow_i_{:04d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'hmdb/frames_tvl1'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{:s}_{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 'something/v1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
        filename_imglist_train = 'something/v1/train_videofolder.txt'
        filename_imglist_val = 'something/v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 'something/v2/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-frames'
        filename_imglist_train = 'something/v2/train_videofolder.txt'
        filename_imglist_val = 'something/v2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'Kinetics/frames_tvl1/'
        filename_imglist_train = 'Kinetics/kinetics400/train_videofolder.txt'
        filename_imglist_val = 'Kinetics/kinetics400/val_videofolder.txt'
        prefix = 'flow_i_{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf11': return_ucf11, 'ucf11-in': return_ucf11_in, 'ucf11-out': return_ucf11_out,
                   'ucf50': return_ucf50, 'ucf50-in': return_ucf50_in, 'ucf50-out': return_ucf50_out,
                   'ucf101': return_ucf101, 'ucf101-in': return_ucf101_in, 'ucf101-out': return_ucf101_out,
                   'hmdb51': return_hmdb51, 'hmdb51-in': return_hmdb51_in, 'hmdb51-out': return_hmdb51_out,
                   'kinetics': return_kinetics }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
