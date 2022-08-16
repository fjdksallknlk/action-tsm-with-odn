#!/bin/bash

# choice = [ucf11, ucf50, ucf101, hmdb51]
dataset=ucf50

echo "------ Starting scripts with arguments ------"
if [ $# -gt 0 ]
then
    if [ $# == 1 ]
    then
        dataset=$1
    elif [ $# -gt 2 ]
    then
        dataset=$1
        echo "extra argv ......"
    fi
else
    echo "Using the default arguments"
fi

echo "Dataset: ${dataset}"
echo "---------------------------------------------"

inputfiles=../../output/TSM_${dataset}-in_RGB_resnet50_avg_segment8_e50/res/res.txt,../../output/TSM_${dataset}-in_RGB_resnet50_avg_segment8_e50/triplet/res.txt
# inputfiles=../../output/TSM_CPL_nlldce_${dataset}-in_RGB_resnet50_avg_segment8_e50/res/res.txt,../../output/TSM_CPL_nlldce_${dataset}-in_RGB_resnet50_avg_segment8_e50/triplet/res.txt

python ./cal_draw_auc.py \
    --inputfiles=${inputfiles}