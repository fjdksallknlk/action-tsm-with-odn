ROOT=/home/shuyu/github/temporal-shift-module

step2=1
#### step 2: compute mavs for openmax ####
FeaturePath=${ROOT}/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/features/train
MavPath=${ROOT}/output/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/mav/
ClassLabel=./static/ucf101_class.txt

if [ ${step2} -eq 1 ]; then
    echo "**** Step2: Compute mavs for openmax ****"
    python ./compute_mav.py \
        --feature_path=${FeaturePath} \
        --save_path=${MavPath} \
        --classlabel=${ClassLabel}
fi