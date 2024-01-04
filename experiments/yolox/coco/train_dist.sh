#!/usr/bin/env bash
set -x

ROOT=../../..
export PYTHONPATH=${ROOT}:${PYTHONPATH}

export JOBLIB_TEMP_FOLDER=/dev/shm
export TMPDIR=/dev/shm
#export CUDA_VISIBLE_DEVICES='0,1'

EXP_NAME=${1-baseline_yolox_s_b32}
CONFIG_FILE=${2-${ROOT}/configs/yolox/yolox_s_4x32_300e_coco.py}
GPUS=${3-'0,1,2,3'}
DIST_PORT=${4-29500}

# get gpu-nums
gpu_ids=(${GPUS//,/ })
ngpus=${#gpu_ids[@]}
export CUDA_VISIBLE_DEVICES=${GPUS}


OUTPUT_DIR=output/${EXP_NAME}_${ngpus}gpus
# for safety
#if [ -d ${OUTPUT_DIR} ]; then
#    echo "job:" ${EXP_NAME} "exists before, 3 seconds for your withdrawing..."
#    sleep 3
#    rm -rf ${OUTPUT_DIR}
#fi

mkdir -p ${OUTPUT_DIR}

echo 'start training:' ${EXP_NAME} 'config:' ${CONFIG_FILE}

python -m torch.distributed.launch --nproc_per_node=${ngpus} --master_port=${DIST_PORT} \
${ROOT}/tools/train.py \
    ${CONFIG_FILE} \
    --work-dir=${OUTPUT_DIR} \
    --launcher pytorch \
    --auto-resume \
    2>&1 | tee ${OUTPUT_DIR}/train_dist.log

echo ${EXP_NAME} 'done.'
#    --resume-from output/baseline_ssd300_coco2017_4gpus/latest.pth \

