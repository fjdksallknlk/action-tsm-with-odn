#!/bin/bash

# choice = [ucf11, ucf50, ucf101, hmdb51]
dataset=ucf50
num_classes=50

echo "------ Starting scripts with arguments ------"
if [ $# -gt 0 ]
then
    if [ $# == 1 ]
    then
        dataset=$1
    elif [ $# == 2 ]
    then
        dataset=$1
        num_classes=$2
    elif [ $# -gt 2 ]
    then
        dataset=$1
        num_classes=$2
        echo 'extra argvs ......'
    fi
else
    echo "Using the default arguments"
fi

echo "Dataset: ${dataset}"
echo "Num_classes: ${num_classes}"
echo "---------------------------------------------"

model=TSM_${dataset}-in_RGB_resnet50_avg_segment8_e50
ClassLabel=./static/${dataset}-in_class.txt

# for triplet options: [triplet, multiple, triplet_multiple]
ThresType=triplet

# for openmax
NUM_CLASSES=${num_classes}
Sigma=14

#### step 1: extract features ####
step1=1
#### step 2: compute mavs for openmax ####
step2=1
#### step 3: compute dis for openmax ####
step3=1
#### step 4: compute openmax scores ####
step4=1
#### step 5: compute thres for triplet threshold ####
step5=1
#### step 6: eval triplet threshold method ####
step6=1
#### step 7: eval openmax method ####
step7=1

# static
ROOT=/home/shuyu/github/temporal-shift-module

FeaturePath=${ROOT}/checkpoint/${model}/features/train
TestFeaturePath=${ROOT}/checkpoint/${model}/features/test/

MavPath=${ROOT}/output/${model}/mav/
DisPath=${ROOT}/output/${model}/dis/
OpenmaxPath=${ROOT}/output/${model}/os/

ThresPath=${ROOT}/output/${model}/thres/
ThresFile=${ROOT}/output/${model}/thres/thres.mat

TripletResult=${ROOT}/output/${model}/triplet/
OpenmaxResult=${ROOT}/output/${model}/res/

#### step 1: extract features ####


#### step 2: compute mavs for openmax ####
if [ ${step2} -eq 1 ]; then
    echo "**** Step2: Compute mavs for openmax ****"
    python ./compute_mav.py \
        --feature_path=${FeaturePath} \
        --save_path=${MavPath} \
        --classlabel=${ClassLabel}
fi


#### step 3: compute dis for openmax ####
if [ ${step3} -eq 1 ]; then
    echo "**** Step3: Compute dis for openmax ****"
    python ./compute_dis.py \
        --mav_path=${MavPath} \
        --feature_path=${FeaturePath} \
        --save_path=${DisPath} \
        --classlabel=${ClassLabel}
fi


#### step 4: compute openmax scores ####
if [ ${step4} -eq 1 ]; then
    echo "**** Step4: Compute openmax scores ****"
    python ./compute_openmax.py \
        --mean_files_path=${MavPath} \
        --category_name=${ClassLabel} \
        --video_feat_path=${TestFeaturePath} \
        --distance_path=${DisPath} \
        --save_path=${OpenmaxPath} \
        --num_classes=${NUM_CLASSES} \
        --sigma=${Sigma}
fi


#### step 5: compute thres for triplet threshold ####
if [ ${step5} -eq 1 ]; then
    echo "**** Step5: Compute thres for triplet threshold method ****"
    python ./compute_thres.py \
        --feature_path=${FeaturePath} \
        --save_path=${ThresPath} \
        --classlabel=${ClassLabel} \
        --thres_type=${ThresType}
fi


#### step 6: eval triplet threshold method ####
if [ ${step6} -eq 1 ]; then
    echo "**** Step6: Eval triplet threshold method ****"
    python ./multi_process_loop_argv_thres.py \
        --save_path=${TripletResult} \
        --feature_path=${TestFeaturePath} \
        --classlabel=${ClassLabel} \
        --thres_file=${ThresFile} \
        --thres_type=${ThresType}
fi


#### step 7: eval openmax method ####
if [ ${step7} -eq 1 ]; then
    python ./multi_process_loop_openmax_rho.py \
        --openmax_path=${OpenmaxPath} \
        --save_path=${OpenmaxResult} \
        --classlabel=${ClassLabel} \
        --numclass=${NUM_CLASSES} \
        --sigma=${Sigma}
fi
