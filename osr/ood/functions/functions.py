# -*- coding:utf8 -*-
import os
import numpy as np

import torch
from torch.autograd import Variable

from dataset import data_loader
from lib import lib_generation
from lib import lib_regression
from utils import callog

from sklearn.linear_model import LogisticRegressionCV


def measure_Baseline(model, modality, test_loader, ood_dist_list, transform, cfg):
    '''
    baseline method softmax score
    '''
    res_list = []
    magnitude, temperature = 0, 1

    # calculate score
    lib_generation.get_posterior(model, test_loader, magnitude, temperature, cfg.output_dir, True)
    for ood_dist in ood_dist_list:
        ood_test_loader = data_loader.getDataSet(data_type="outData", dataset=ood_dist,
                                                        modality=modality, batch_size=cfg.batch_size, 
                                                        num_segments=cfg.num_segments, transform=transform, workers=cfg.workers, 
                                                        dense_sample=cfg.dense_sample, twice_sample=cfg.twice_sample)
        print('Out-distribution: %s'% ood_dist)
        lib_generation.get_posterior(model, ood_test_loader, magnitude, temperature, cfg.output_dir, False)
        test_results = callog.metric(cfg.output_dir, ['PoT'])
        res_list.append(test_results)

    return res_list


def measure_ODIN(model, modality, test_loader, ood_dist_list, transform, cfg):
    '''
    ODIN method: softmax score with temperature and magnitude
    '''
    M_list = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
    T_list = [1, 10, 100, 1000]
    ODIN_best_tnr = [0, 0, 0]
    ODIN_best_results = [0 , 0, 0]
    ODIN_best_temperature = [-1, -1, -1]
    ODIN_best_magnitude = [-1, -1, -1]
    for T in T_list:
        for m in M_list:
            magnitude = m
            temperature = T
            lib_generation.get_posterior(model, test_loader, magnitude, temperature, cfg.output_dir, True)
            out_count = 0
            print('Temperature: %s / noise: %s' % (str(temperature), str(magnitude)))
            for ood_dist in ood_dist_list:
                ood_test_loader = data_loader.getDataSet(data_type="outData", dataset=ood_dist,
                                                        modality=modality, batch_size=cfg.batch_size, 
                                                        num_segments=cfg.num_segments, transform=transform, workers=cfg.workers, 
                                                        dense_sample=cfg.dense_sample, twice_sample=cfg.twice_sample)
                print('Out-distribution: %s'% ood_dist)
                lib_generation.get_posterior(model, ood_test_loader, magnitude, temperature, cfg.output_dir, False)
                if temperature == 1 and magnitude == 0:
                    # Baseline case continue
                    continue
                else:
                    val_results = callog.metric(cfg.output_dir, ['PoV'])
                    if ODIN_best_tnr[out_count] < val_results['PoV']['TNR']:
                        ODIN_best_tnr[out_count] = val_results['PoV']['TNR']
                        ODIN_best_results[out_count] = callog.metric(cfg.output_dir, ['PoT'])
                        ODIN_best_temperature[out_count] = temperature
                        ODIN_best_magnitude[out_count] = magnitude
                out_count += 1

    ODIN_best_arguments = {'ODIN_best_temperature': ODIN_best_temperature,
            'ODIN_best_magnitude': ODIN_best_magnitude}

    return ODIN_best_results, ODIN_best_arguments


def measure_Mahalanobis(model, modality, train_loader, test_loader, ood_dist_list, transform, cfg):
    # set information about feature extaction
    model.eval()
    temp_x = torch.rand(4,3,224,224).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance')
    sample_mean, precision = lib_generation.sample_estimator(model, cfg.num_classes, feature_list, train_loader)

    print('get Mahalanobis scores')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('Noise: %s'% str(magnitude))
        for i in range(num_output):
            M_in = lib_generation.get_Mahalanobis_score(model, test_loader, cfg.num_classes, cfg.output_dir, \
                                                        True, sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

        for ood_dist in ood_dist_list:
            ood_test_loader = data_loader.getDataSet(data_type="outData", dataset=ood_dist,
                                                        modality=modality, batch_size=cfg.batch_size, 
                                                        num_segments=cfg.num_segments, transform=transform, workers=cfg.workers, 
                                                        dense_sample=cfg.dense_sample, twice_sample=cfg.twice_sample)
            print('Out-distribution: %s'% ood_dist)
            for i in range(num_output):
                M_out = lib_generation.get_Mahalanobis_score(model, ood_test_loader, cfg.num_classes, cfg.output_dir, \
                                                             False, sample_mean, precision, i, magnitude)
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
            file_name = os.path.join(cfg.output_dir, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), cfg.in_dataset, ood_dist))
            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)


    print('Mahalanobis detector')

    # initial setup
    # dataset_list = ['cifar10', 'cifar100', 'svhn']
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', 'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']

    # train and measure the performance of Mahalanobis detector
    list_best_results, list_best_results_index = [], []
    # for dataset in dataset_list:
    dataset = cfg.in_dataset
    print('In-distribution: %s'% dataset)
    outf = cfg.output_dir
    list_best_results_out, list_best_results_index_out = [], []
    for out in ood_dist_list:
        print('Out-of-distribution: %s'% out)
        best_tnr, best_result, best_index = 0, 0, 0
        for score in score_list:
            total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf)
            X_val, Y_val, X_test, Y_test = lib_regression.block_split(total_X, total_Y, cfg.in_dataset, out)
            X_train = np.concatenate((X_val[:50], X_val[100:150]))
            Y_train = np.concatenate((Y_val[:50], Y_val[100:150]))
            X_val_for_test = np.concatenate((X_val[50:100], X_val[150:]))
            Y_val_for_test = np.concatenate((Y_val[50:100], Y_val[150:]))
            lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
            y_pred = lr.predict_proba(X_train)[:, 1]
            y_pred = lr.predict_proba(X_val_for_test)[:, 1]
            results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
            if best_tnr < results['TMP']['TNR']:
                best_tnr = results['TMP']['TNR']
                best_index = score
                best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)
        list_best_results_out.append(best_result)
        list_best_results_index_out.append(best_index)
    list_best_results.append(list_best_results_out)
    list_best_results_index.append(list_best_results_index_out)

    # print the results
    count_in = 0
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']

    for in_list in list_best_results:
        print('in_distribution: ' + dataset + '==========')
        out_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        if dataset == 'svhn':
            out_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ out_list[count_out])
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
            print('Input noise: ' + list_best_results_index[count_in][count_out])
            print('')
            count_out += 1
        count_in += 1
