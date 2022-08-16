# -*- coding：utf-8 -*-


# choice = [ucf11, ucf50, ucf101, hmdb51]
dataset=ucf50-in

# mode choice = [Baseline, ODIN, Mahalanobis]
mode=Baseline
# mode=ODIN
# mode=Mahalanobis

num_class=25

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

MODEL=../../checkpoint/TSM_${dataset}_RGB_resnet50_avg_segment8_e50/ckpt.pth.tar

# Mahalanobis 时需要加before_softmax，其余不加
CUDA_VISIBLE_DEVICES=${GPU_IDs} python ./OOD_main.py \
    --mode=${mode} \
    --in_dataset=${dataset} \
    --num_classes=${num_class} \
    --weights=${MODEL} \
    --num_segments=4 \
    --test_crops=1 \
    --batch_size=32 \
    --set_seed=True \
    # --before_softmax