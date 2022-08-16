#!/bin/bash

# choice = [ucf11, ucf50, ucf101, hmdb51]
dataset=ucf50
# choice = [resnet50, resnet101, mobilenetv2, efficientnet-b0]
mode=test
num_class=50
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
        mode=$2
    elif [ $# -gt 2 ]
    then
        dataset=$1
        mode=$2
        num_class=$3
        GPU_ID=()
        shift 3
        for gpu in $@; do
            GPU_ID[${#GPU_ID[@]}]=${gpu}
        done
    fi
else
    echo "Using the default arguments"
fi

echo "Dataset: ${dataset}"
echo "Mode: ${mode}"
echo "Num_Class: ${num_class}"
echo "GPU_IDs: ${GPU_ID[@]}"
echo "---------------------------------------------"

GPU_IDs=''
for i in ${GPU_ID[@]};do
  GPU_IDs=$GPU_IDs","$i;
done
GPU_IDs=${GPU_IDs:1}

model=checkpoint/TSM_${dataset}-in_RGB_resnet50_avg_segment8_e50/ckpt.pth.tar

# test TSM
CUDA_VISIBLE_DEVICES=${GPU_IDs} python test.py ${dataset} \
    --weights=${model} \
    --test_segments=4 --test_crops=1 \
    --batch_size=64 \
    --extract \
    --mode=${mode} \
    --num_class=${num_class} \
    --del_features \
    # --softmax
