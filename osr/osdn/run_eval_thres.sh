ROOT=/home/shuyu/github/temporal-shift-module

step6=1
#### step 6: eval triplet threshold method ####
ClassLabel=./static/ucf50-in_class.txt
TestFeaturePath=${ROOT}/checkpoint/TSM_ucf50-in_RGB_resnet50_avg_segment8_e50/features/test/

ThresType=triplet

ThresPath=${ROOT}/output/TSM_ucf50-in_RGB_resnet50_avg_segment8_e50/thres/thres.mat


ResultPath=${ROOT}/output/TSM_ucf50-in_RGB_resnet50_avg_segment8_e50/triplet/

if [ ${step6} -eq 1 ]; then
    echo "**** Step6: Eval triplet threshold method ****"
    python ./multi_process_loop_argv_thres.py \
        --save_path=${ResultPath} \
        --feature_path=${TestFeaturePath} \
        --classlabel=${ClassLabel} \
        --thres_file=${ThresPath} \
        --thres_type=${ThresType}
fi