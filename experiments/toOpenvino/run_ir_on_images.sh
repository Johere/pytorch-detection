#/bin/bash
ROOT=../..
DATASET_ROOT=/mnt/disk1/data_for_linjiaojiao/datasets  # 2080ti
# DATASET_ROOT=/mnt/disk1/data_for_jiaojiao/datasets   # v100

IR_MODEL_FILE=${1-"output/sddlite_exp2_4gpus_ov-2021.4.582/FP16/latest.pth.xml"}
METHOD=${2-"ssdlite_barrier"}  # 'barrier106', 'ssdlite', 'ssdlite_barrier'
THRESH=${3-0}
VIS_RATIO=${4-0}
IMAGE_DIR=${5-"/mnt/disk3/data_for_linjiaojiao/datasets/HCE_test/road_barrier_1920_1080_fps30"}  # bit, uadetrac
IMAGE_FLAG=${IMAGE_DIR##*/}

tmp0=${IR_MODEL_FILE%/*}  # output/sddlite_exp2_4gpus_ov-2021.4.582/FP16
tmp1=${tmp0%/*}  # output/sddlite_exp2_4gpus_ov-2021.4.582
EXP_NAME=${tmp1##*/}  # sddlite_exp2_4gpus_ov-2021.4.582

source /mnt/disk1/data_for_linjiaojiao/intel/openvino_2021.4.582/bin/setupvars.sh
export PYTHONPATH=${ROOT}:${PYTHONPATH}

if [[ ! -d "results" ]]; then
  mkdir results
fi


if [[ ! -d "logs" ]]; then
  mkdir logs
fi


JOB_NAME=profile_ir_${EXP_NAME}_thr${THRESH}_${IMAGE_FLAG}
OUTPUT_DIR=results/${JOB_NAME}
echo 'start: ' ${JOB_NAME} 'on image_dir:' ${IMAGE_DIR}
echo output_dir: ${OUTPUT_DIR}

python -u ${ROOT}/tools/openvino/run_vehicle_detection.py \
    --ir_xml ${IR_MODEL_FILE} \
    -m ${METHOD} \
    --images_dir ${IMAGE_DIR} \
    -o ${OUTPUT_DIR} \
    -v ${VIS_RATIO} \
    --thresh ${THRESH} \
    2>&1 | tee logs/${JOB_NAME}.log
