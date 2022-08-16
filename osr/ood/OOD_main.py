# -*- coding:utf8 -*-

import os
import numpy as np
import argparse

import torch
from torch.autograd import Variable
from torchvision import transforms

from dataset import data_loader
from functions.functions import measure_Baseline
from functions.functions import measure_ODIN
from functions.functions import measure_Mahalanobis
from utils.utils import str2bool, str2list
from utils.utils import may_mkdir

ROOT_PATH = '../../'
import sys
sys.path.append(ROOT_PATH)

from ops.models import TSN
from ops.transforms import *


def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None


class Config():
    '''
    config class
    argparse and config setting
    '''
    def __init__(self):
        parser = argparse.ArgumentParser(description='Mahalanobis detector')

        # method
        parser.add_argument('--mode', type=str, default='baseline',
                choices=['Baseline', 'ODIN', 'Mahalanobis'],
                help='The mode[method], default baseline.')
        parser.add_argument('--output_dir', type=str, default=ROOT_PATH + '/output/ood/',
                help='The dir to save the tmp result, softmax scores and Mahalanobis scores, default ./output/')

        # data
        parser.add_argument('--in_dataset', type=str, default='ucf101',
                help='The in-distribution dataset, default cifar10, all the other dataset are regarded as ood.')
        parser.add_argument('--num_classes', type=int, default=101,
                help='The class number of the in-distribution data, default 101.')

        # model
        parser.add_argument('--weights', type=str, default=None)
        parser.add_argument('--num_segments', type=int, default=5)
        parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
        parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
        parser.add_argument('--full_res', default=False, action="store_true",
                            help='use full resolution 256x256 for test as in Non-local I3D')
        parser.add_argument('--test_crops', type=int, default=1)
        parser.add_argument('--coeff', type=str, default=None)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                            help='number of data loading workers (default: 8)')

        # devices and env
        parser.add_argument('--gpus', type=str2list, default=None)
        parser.add_argument('--set_seed', type=str2bool, default=True,
                help='If set seed, default True.')

        # for true test
        parser.add_argument('--test_file', type=str, default=None)
        parser.add_argument('--csv_file', type=str, default=None)

        parser.add_argument('--before_softmax', default=False, action="store_true", help='use softmax')

        parser.add_argument('--max_num', type=int, default=-1)
        parser.add_argument('--input_size', type=int, default=224)
        parser.add_argument('--crop_fusion_type', type=str, default='avg')
        
        parser.add_argument('--img_feature_dim',type=int, default=256)
        parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
        parser.add_argument('--pretrain', type=str, default='imagenet')

        args = parser.parse_args()

        # model
        self.weights = args.weights
        self.batch_size = args.batch_size
        self.full_res = args.full_res
        self.test_crops = args.test_crops
        self.workers = args.workers
        self.dense_sample = args.dense_sample
        self.twice_sample = args.twice_sample

        # method
        self.mode = args.mode
        self.output_dir = args.output_dir

        # data
        self.in_dataset = args.in_dataset
        self.num_classes = args.num_classes

        # devices
        self.gpus = args.gpus
        self.set_seed = args.set_seed

        # for true test
        self.num_segments = args.num_segments
        self.crop_fusion_type = args.crop_fusion_type
        self.img_feature_dim = args.img_feature_dim
        self.pretrain = args.pretrain
        self.before_softmax = args.before_softmax

        # gen args
        weight = args.weights.split('/')[3]
        self.output_dir = os.path.join(args.output_dir, weight, args.in_dataset)
        may_mkdir(self.output_dir)



def main():
    '''
    main function
    '''
    cfg = Config()
    import pprint
    print('-'*60)
    pprint.pprint(cfg.__dict__)
    print('-'*60)

    if cfg.set_seed == True:
        torch.cuda.manual_seed(0)
    # torch.cuda.set_device(cfg.gpus)

    if cfg.in_dataset == 'ucf11':
        ood_dist_list = ['hmdb51']
    elif cfg.in_dataset == 'ucf50':
        ood_dist_list = ['hmdb51']
    elif cfg.in_dataset == 'ucf101':
        ood_dist_list = ['hmdb51']
    elif cfg.in_dataset == 'hmdb51':
        # in_dataset hmdb51
        ood_dist_list = ['ucf11', 'ucf50', 'ucf101']
    elif cfg.in_dataset == 'ucf11-in':
        ood_dist_list = ['ucf11-out']
    elif cfg.in_dataset == 'ucf50-in':
        ood_dist_list = ['ucf50-out']
    elif cfg.in_dataset == 'ucf101-in':
        ood_dist_list = ['ucf101-out']
    elif cfg.in_dataset == 'hmdb51-in':
        ood_dist_list = ['hmdb51-out']

    weights = cfg.weights

    is_shift, shift_div, shift_place = parse_shift_option_from_log_name(weights)
    print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))

    if 'RGB' in weights:
        modality = 'RGB'
    else:
        modality = 'Flow'
    arch = weights.split('TSM_')[1].split('_')[2]

    net = TSN(cfg.num_classes, cfg.num_segments,
              modality,
              base_model=arch,
              consensus_type=cfg.crop_fusion_type,
              before_softmax=cfg.before_softmax,
              img_feature_dim=cfg.img_feature_dim,
              pretrain=cfg.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local='_nl' in weights,
            )
    
    if 'tpool' in weights:
        from ops.temporal_shift import make_temporal_pool
        make_temporal_pool(net.base_model, test_segments)  # since DataParallel

    checkpoint = torch.load(weights)
    checkpoint = checkpoint['state_dict']

    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
                    }
    for k, v in replace_dict.items():
        if k in base_dict:
            base_dict[v] = base_dict.pop(k)

    net.load_state_dict(base_dict)

    input_size = net.scale_size if cfg.full_res else net.input_size
    if cfg.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(input_size),
        ])
    elif cfg.test_crops == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(input_size, net.scale_size, flip=False)
        ])
    elif cfg.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size, flip=False)
        ])
    elif cfg.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(cfg.test_crops))

    transform = torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])


    # net = torch.nn.DataParallel(net.cuda())
    net.cuda()
    net.eval()

    # data
    print('load in-distribution data: %s' % cfg.in_dataset)
    train_loader, test_loader = data_loader.getDataSet(data_type="inData", dataset=cfg.in_dataset,
                                                        modality=modality, batch_size=cfg.batch_size, 
                                                        num_segments=cfg.num_segments, transform=transform, workers=cfg.workers, 
                                                        dense_sample=cfg.dense_sample, twice_sample=cfg.twice_sample)

    # measure the performance
    if cfg.mode == 'Baseline':
        results = measure_Baseline(net, modality, test_loader, ood_dist_list, transform, cfg)
    elif cfg.mode == 'ODIN':
        results, arguments = measure_ODIN(net, modality, test_loader, ood_dist_list, transform, cfg)
    elif cfg.mode == 'Mahalanobis':
        measure_Mahalanobis(net, modality, train_loader, test_loader, ood_dist_list, transform, cfg)
    else:
        raise ValueError('The mode[method] %s is not support.'% cfg.mode)

    if cfg.mode != 'Mahalanobis':
        # print the results
        mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
        print('Method of %s: in_distribution dataset: %s ========'% (cfg.mode, cfg.in_dataset))
        count_out = 0

        import pprint
        print('-'*60)
        pprint.pprint(results)
        print('-'*60)

        for res in results:
            print('out_distribution: %s'% ood_dist_list[count_out])
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*res['PoT']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*res['PoT']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*res['PoT']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*res['PoT']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*res['PoT']['AUOUT']), end='')
            if cfg.mode == 'ODIN':
                print('temperature: %s' % str(arguments['ODIN_best_temperature'][count_out]))
                print('magnitude: %s' % str(arguments['ODIN_best_magnitude'][count_out]))
            print('')
            count_out += 1
            break


if __name__ == '__main__':
    main()
