
# set run flag
step1=1
step2=1
step3=1
step4=1
step5=1
step6=1
step7=1

#mode spatial or temporal
#mode=spatial
mode=temporal


#method=base
#method=test
#method=gcpl
method=prototype

ThresType=triplet
#ThresType=multiple
#ThresType=triplet_multiple

Sigma=15

# set argvs for ucf11
Data=UCF11
dataset=UCF11
ClassLabel=./static/${Data}_known_classlist
NUM_CLASSES=6

## set argvs for ucf50
#Data=UCF50
#dataset=UCF50
#ClassLabel=./static/${Data}_known_classlist
#NUM_CLASSES=25

## set argvs for ucf101
#Data=UCF101_split1
#dataset=UCF101
#ClassLabel=./static/${Data}_known_classlist
#NUM_CLASSES=50

## set argvs for hmdb51
#dataset=HMDB51
#Data=HMDB51
#ClassLabel=./static/${Data}_known_classlist
#NUM_CLASSES=25



#### step 1: extract features ####
Train_Feat_path=../../scripts/cnns/${Data}/${dataset}_${mode}_inception_resnet_v2_299_${method}_init/features_train
Train_Save_path_name=${Data}_${mode}_irs_${method}_train

Test_Feat_path=../../scripts/cnns/${Data}/${dataset}_${mode}_inception_resnet_v2_299_${method}_init/features_test
Test_Save_path_name=${Data}_${mode}_irs_${method}_testlist

if [ ${step1} -eq 1 ]; then
    echo "**** Step1.1: Extract train feat with score ****"
    python ./extract_feature.py ${Train_Feat_path} ${Train_Save_path_name} ${dataset}
    echo "**** Step1.2: Extract test feat with score ****"
    python ./extract_feature.py ${Test_Feat_path} ${Test_Save_path_name} ${dataset}
fi


#### step 2: compute mavs for openmax ####
FeaturePath=./features/${Data}_${mode}_irs_${method}_train/
MavPath=./output/mav/${Data}_${mode}_irs_${method}/

if [ ${step2} -eq 1 ]; then
    echo "**** Step2: Compute mavs for openmax ****"
    python ./mav_compute.py \
        --feature_path=${FeaturePath} \
        --save_path=${MavPath} \
        --classlabel=${ClassLabel}
fi


#### step 3: compute dis for openmax ####
DisPath=./output/dis/${Data}_${mode}_irs_${method}

if [ ${step3} -eq 1 ]; then
    echo "**** Step3: Compute dis for openmax ****"
    python ./dis_compute.py \
        --mav_path=${MavPath} \
        --feature_path=${FeaturePath} \
        --save_path=${DisPath} \
        --classlabel=${ClassLabel}
fi


#### step 4: compute openmax scores ####
TestFeaturePath=./features/${Data}_${mode}_irs_${method}_testlist/
OpenmaxPath=./output/openmax/${Data}_${mode}_irs_${method}

if [ ${step4} -eq 1 ]; then
    echo "**** Step4: Compute openmax scores ****"
    python ./compute_openmax.py \
        --mean_files_path=${MavPath} \
        --category_name=${ClassLabel} \
        --video_feat_path=${TestFeaturePath} \
        --distance_path=${DisPath} \
        --save_path=${OpenmaxPath} \
        --numclasses=${NUM_CLASSES} \
        --sigma=${Sigma}
fi


#### step 5: compute thres for triplet threshold ####
ThresPath=./output/thres/${Data}_${mode}_irs_${method}

if [ ${step5} -eq 1 ]; then
    echo "**** Step5: Compute thres for triplet threshold method ****"
    python ./thres_compute.py \
        --feature_path=${FeaturePath} \
        --save_path=${ThresPath} \
        --classlabel=${ClassLabel} \
        --thres_type=${ThresType}
fi


#### step 6: eval triplet threshold method ####
ResultPath=./output/result/thres/${Data}_${mode}_irs_${method}

if [ ${step6} -eq 1 ]; then
    echo "**** Step6: Eval triplet threshold method ****"
    python ./multi_process_loop_argv_thres.py \
        --save_path=${ResultPath} \
        --feature_path=${TestFeaturePath} \
        --classlabel=${ClassLabel} \
        --thres_file=${ThresPath} \
        --thres_type=${ThresType}
fi


#### step 7: eval openmax method ####
ResultPath=./output/result/openmax/${Data}_${mode}_irs_${method}

if [ ${step7} -eq 1 ]; then
    python ./eval_openmax.py \
        --openmax_path=${OpenmaxPath} \
        --save_path=${ResultPath} \
        --classlabel=${ClassLabel} \
        --numclass=${NUM_CLASSES} \
        --sigma=${Sigma}
fi
