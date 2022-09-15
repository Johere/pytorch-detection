#/bin/bash
ROOT=../..

IR_MODEL_FILE=${1-"output/sddlite_exp2_4gpus_ov-2021.4.582/FP16/latest.pth.xml"}
QUANT_ALGORITHM=${2-"default"}  # "default"
INPUT_SIZE=${3-320}
IMAGE_DIR=${4-"/mnt/disk3/data_for_linjiaojiao/datasets/BITVehicle/cropped_images"}

tmp0=${IR_MODEL_FILE%/*}  # output/sddlite_exp2_4gpus_ov-2021.4.582/FP16
tmp1=${tmp0%/*}  # output/sddlite_exp2_4gpus_ov-2021.4.582
EXP_NAME=${tmp1##*/}  # sddlite_exp2_4gpus_ov-2021.4.582

source /mnt/disk1/data_for_linjiaojiao/intel/openvino_2021.4.582/bin/setupvars.sh
export PYTHONPATH=${ROOT}:${PYTHONPATH}


if [[ ! -d "logs" ]]; then
  mkdir logs
fi


JOB_NAME=quanti8_ir_${EXP_NAME}_${QUANT_ALGORITHM}
echo 'start: ' ${JOB_NAME}

python -u ${ROOT}/tools/openvino/quant_i8.py \
    --ir_xml ${IR_MODEL_FILE} \
    -m ${QUANT_ALGORITHM} \
    --images_dir ${IMAGE_DIR} \
    --input_size ${INPUT_SIZE} \
    2>&1 | tee logs/${JOB_NAME}.log
