ROOT=/home/shuyu/github/temporal-shift-module

step7=1
#### step 7: eval openmax method ####
ClassLabel=./static/ucf101_class.txt
TestFeaturePath=${ROOT}/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/features/test/

OpenmaxPath=${ROOT}/output/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/os/

NUM_CLASSES=101
Sigma=15

ResultPath=${ROOT}/output/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50/res/

if [ ${step7} -eq 1 ]; then
    python ./eval_openmax.py \
        --openmax_path=${OpenmaxPath} \
        --save_path=${ResultPath} \
        --classlabel=${ClassLabel} \
        --numclass=${NUM_CLASSES} \
        --sigma=${Sigma}
fi
