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
OUTPUT_DIR=results/test_${EXP_NAME}

echo 'start testing:' ${EXP_NAME} 'config:' ${CONFIG_FILE} 'checkpoint: ' ${CHECKPOINT}
python -u ${ROOT}/tools/test.py ${CONFIG_FILE} ${CHECKPOINT} \
    --save-dir ${OUTPUT_DIR} \
    --format-only \
    --cfg-options \
    data.test.ann_file=/mnt/disk1/data_for_linjiaojiao/datasets/Radical/images/radical_50m_collections_ford_edge_list_fps30.txt \
    data.test.img_prefix=/mnt/disk1/data_for_linjiaojiao/datasets/Radical/images \
    data.test.type=CustomListDataset \
    2>&1 | tee logs/test_${EXP_NAME}.log

echo ${EXP_NAME} 'done.'

# --format-only \
# --cfg-options \
# data.test.ann_file=/mnt/disk1/data_for_linjiaojiao/datasets/Radical/images/radical_50m_collections_ford_edge_list_fps30.txt \
# data.test.img_prefix=/mnt/disk1/data_for_linjiaojiao/datasets/Radical/images \

# show coco results
# --show --show-dir ${OUTPUT_DIR} \
