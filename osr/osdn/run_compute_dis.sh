ROOT=/home/shuyu/github/temporal-shift-module

step3=1
#### step 3: compute dis for openmax ####
FeaturePath=${ROOT}/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/features/train
MavPath=${ROOT}/output/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/mav/
ClassLabel=./static/ucf101_class.txt

DisPath=${ROOT}/output/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/dis/

if [ ${step3} -eq 1 ]; then
    echo "**** Step3: Compute dis for openmax ****"
    python ./compute_dis.py \
        --mav_path=${MavPath} \
        --feature_path=${FeaturePath} \
        --save_path=${DisPath} \
        --classlabel=${ClassLabel}
fi
