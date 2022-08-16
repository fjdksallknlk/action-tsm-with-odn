#!/bin/bash

# choice = [ucf11, ucf50, ucf101, hmdb51]
dataset=ucf50
# choice = [resnet50, resnet101, mobilenetv2, efficientnet-b0]
arch=resnet50
GPU_ID=(0 1 2 3)

echo "------ Starting scripts with arguments ------"
if [ $# -gt 0 ]
then
    if [ $1 == "-g" ]
    then
        GPU_ID=()
        shift 1
        for gpu in $@; do
            GPU_ID[${#GPU_ID[@]}]=${gpu}
        done
    elif [ $# == 1 ]
    then
        dataset=$1
    elif [ $# == 2 ]
    then
        dataset=$1
        arch=$2
    elif [ $# -gt 2 ]
    then
        dataset=$1
        arch=$2
        GPU_ID=()
        shift 2
        for gpu in $@; do
            GPU_ID[${#GPU_ID[@]}]=${gpu}
        done
    fi
else
    echo "Using the default arguments"
fi

echo "Dataset: ${dataset}"
echo "Arch: ${arch}"
echo "GPU_IDs: ${GPU_ID[@]}"
echo "---------------------------------------------"

GPU_IDs=''
for i in ${GPU_ID[@]};do
  GPU_IDs=$GPU_IDs","$i;
done
GPU_IDs=${GPU_IDs:1}

# test TSM
model=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth

# test TSM
CUDA_VISIBLE_DEVICES=${GPU_IDs} python test_models.py ${dataset} \
    --weights=${model} \
    --test_segments=8 --test_crops=1 \
    --batch_size=64
