#!/usr/bin/env bash
set -x

ROOT=../../..
export PYTHONPATH=${ROOT}:${PYTHONPATH}

export JOBLIB_TEMP_FOLDER=/dev/shm
export TMPDIR=/dev/shm
#export CUDA_VISIBLE_DEVICES='0,1'
#GPUS=${3-'0'}
#export CUDA_VISIBLE_DEVICES=${GPUS}

EXP_NAME=${1-debug}
default_cfg=$(ls output/${EXP_NAME}/*.py)
CONFIG_FILE=${2-${default_cfg}}
default_ckpt=$(ls output/${EXP_NAME}/best_bbox_*.pth)
CHECKPOINT=${3-${default_ckpt}}
EVAL_FLAG=${4-bbox}         # bbox, mAP

mkdir -p ./logs

echo 'start testing:' ${EXP_NAME} 'config:' ${CONFIG_FILE} 'checkpoint: ' ${CHECKPOINT}
python -u ${ROOT}/tools/test.py ${CONFIG_FILE} ${CHECKPOINT} \
    --eval ${EVAL_FLAG} \
    2>&1 | tee logs/test_${EXP_NAME}.log

echo ${EXP_NAME} 'done.'

# --format-only \
# ${ROOT}/configs/ssd/ssd300_coco.py
# ${ROOT}/open_model_zoo/ssd/ssd300_coco_20210803_015428-d231a06e.pth
#CONFIG_FILE=${2-${ROOT}/configs/ssd/ssd512_coco.py}
#CHECKPOINT=${3-${ROOT}/open_model_zoo/ssd/ssd512_coco_20210803_022849-0a47a1ca.pth}
