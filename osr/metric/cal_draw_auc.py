#! -*- coding=utf-8 -*-
import matplotlib.pyplot as plt
import argparse


def draw_roc(f):
    xy_arr = []
    prev_x = 0
    with open(f, 'r') as fh:
        for ln in fh:
            # ln => FPR: 0.6028 TPR: 0.9171 K_P: 0.8036 K_R: 0.9171 UK_P: 0.6405 UK_R: 0.3972 K_Fscore: 0.8566 UK_Fscore: 0.4904 acc: 0.6810
            items = ln.strip().split(' ')
            if float(items[1]) - prev_x > 0.00001:
                xy_arr.append([float(items[1]), float(items[3]), float(items[17])])
                prev_x = float(items[1])
                if abs(float(items[3]) - float(items[17])) < 0.01:
                    print ("ACC: %.4f" % ((float(items[3]) + float(items[17]))/2))
    x = [_v[0] for _v in xy_arr]
    y1 = [_v[2] for _v in xy_arr]
    y2 = [_v[1] for _v in xy_arr]

    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(x, y1)
    plt.plot(x, y2)


def draw_aupr_k(f):
    pass


def draw_aupr_uk(f):
    pass


def cal(xy_arr):
    xy_arr.sort()

    #计算曲线下面积
    auc = 0
    prev_x = 0
    acc = 0.0
    cnt = 0
    cal_acc = True if len(xy_arr[0]) == 3 else False
    for item in xy_arr:
        x, y = item[0], item[1]
        if x - prev_x > 0.0001:
            auc += (x - prev_x) * y
            prev_x = x
            if cal_acc and abs(item[1] - item[2]) < 0.1:
                acc += ((item[1] + item[2]) / 2)
                cnt += 1
    
    acc = acc/cnt if cnt > 0 else 0

    return auc, acc


def cal_auc(f):
    auroc_arr, auprk_arr, aupruk_arr = [], [], []
    with open(f, 'r') as fh:
        for ln in fh:
            # ln => FPR: 0.6028 TPR: 0.9171 K_P: 0.8036 K_R: 0.9171 UK_P: 0.6405 UK_R: 0.3972 K_Fscore: 0.8566 UK_Fscore: 0.4904 acc: 0.6810
            items = ln.strip().split(' ')
            auroc_arr.append([float(items[1]), float(items[3]), float(items[17])])
            auprk_arr.append([float(items[7]), float(items[5])])
            aupruk_arr.append([float(items[11]), float(items[9])])

    auroc, acc = cal(auroc_arr)
    auprk,_ = cal(auprk_arr)
    aupruk,_ = cal(aupruk_arr)

    print ("AUROC: %.4f AUPRK: %.4f AUPRUK: %.4f ACC: %.4f\n" % (auroc, auprk, aupruk, acc))

    return auroc, auprk, aupruk, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfiles', type=str, default='../../output/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/res/res.txt')
    parser.add_argument('--draw_roc', default=False, action="store_true")
    parser.add_argument('--draw_aupr_k', default=False, action="store_true")
    parser.add_argument('--draw_aupr_uk', default=False, action="store_true")

    args = parser.parse_args()

    input_filelist = args.inputfiles.split(',')

    for f in input_filelist:
        info = f.split('/')
        model, method = info[3], info[4]
        method = 'openmax' if method == 'res' else method
        print ("Model: {}\nMethod: {}".format(model, method))
        auroc, aupr_k, aupr_uk, acc = cal_auc(f)
        if args.draw_roc:
            draw_roc(f)
        if args.draw_aupr_k:
            draw_aupr_k(f)
        if args.draw_aupr_uk:
            draw_aupr_uk(f)
    
    if args.draw_roc or args.draw_aupr_k or args.draw_aupr_uk:
        plt.savefig('./tmp.jpg')
        plt.show()# show the plot on the screen

        plt.close()

    print('Done')
