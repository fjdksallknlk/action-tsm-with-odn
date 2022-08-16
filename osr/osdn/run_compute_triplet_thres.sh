ROOT=/home/shuyu/github/temporal-shift-module

step5=1
#### step 5: compute thres for triplet threshold ####
ClassLabel=./static/ucf101_class.txt
FeaturePath=${ROOT}/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/features/train/

ThresType=triplet

ThresPath=${ROOT}/output/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/thres/

if [ ${step5} -eq 1 ]; then
    echo "**** Step5: Compute thres for triplet threshold method ****"
    python ./compute_thres.py \
        --feature_path=${FeaturePath} \
        --save_path=${ThresPath} \
        --classlabel=${ClassLabel} \
        --thres_type=${ThresType}
fi