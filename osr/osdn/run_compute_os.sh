ROOT=/home/shuyu/github/temporal-shift-module

step4=1
#### step 4: compute openmax scores ####
ClassLabel=./static/ucf101_class.txt
TestFeaturePath=${ROOT}/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/features/test/
MavPath=${ROOT}/output/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/mav/
DisPath=${ROOT}/output/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/dis/

OpenmaxPath=${ROOT}/output/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/os/

NUM_CLASSES=101
Sigma=15

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