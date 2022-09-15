#!/usr/bin/env bash
set -x

ROOT=../../..
export PYTHONPATH=${ROOT}:${PYTHONPATH}

export JOBLIB_TEMP_FOLDER=/dev/shm
export TMPDIR=/dev/shm
#export CUDA_VISIBLE_DEVICES='0,1'
#GPUS=${3-'0'}
#export CUDA_VISIBLE_DEVICES=${GPUS}

EXP_NAME=${1-debug}  # ssd300_uadetrac_4gpus
cfg=$(ls output/${EXP_NAME}/*.py)
CONFIG_FILE=${2-${cfg}}
ckpt=$(ls output/${EXP_NAME}/latest.pth)
CHECKPOINT=${3-${ckpt}}

JOB_NAME=test_UA_DETRAC-val-${EXP_NAME}
#JOB_NAME=test_${EXP_NAME}

mkdir -p ./logs

echo 'start testing:' ${EXP_NAME} 'config:' ${CONFIG_FILE} 'checkpoint: ' ${CHECKPOINT}
python -u ${ROOT}/tools/test.py ${CONFIG_FILE} ${CHECKPOINT} \
    --eval mAP \
    --cfg-options \
    data.test.ann_file=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/val_meta.list \
    data.test.img_prefix=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/images \
    2>&1 | tee logs/${JOB_NAME}.log

echo ${JOB_NAME} 'done.'

# --format-only \
# ${ROOT}/configs/ssd/ssd300_coco.py
# ${ROOT}/open_model_zoo/ssd/ssd300_coco_20210803_015428-d231a06e.pth
#CONFIG_FILE=${2-${ROOT}/configs/ssd/ssd512_coco.py}
#CHECKPOINT=${3-${ROOT}/open_model_zoo/ssd/ssd512_coco_20210803_022849-0a47a1ca.pth}
#    --cfg-options \
#    data.test.ann_file=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/val_meta.list \
#    data.test.img_prefix=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/images \
#    data.test.ann_file=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/test_meta.list \
#    data.workers_per_gpu=0 \
