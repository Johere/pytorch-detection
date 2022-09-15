#/bin/bash
ROOT=../..

IR_MODEL_FILE=${1-"output/sddlite_exp2_4gpus_ov-2021.4.582/FP16/latest.pth.xml"}
METHOD=${2-"ssdlite_barrier"}  # 'barrier106', 'ssdlite', 'ssdlite_barrier'
THRESH=${3-0}
VIS_RATIO=${4-0}
TEST_SET=${5-"bit"}  # bit, uadetrac

tmp0=${IR_MODEL_FILE%/*}  # output/sddlite_exp2_4gpus_ov-2021.4.582/FP16
tmp1=${tmp0%/*}  # output/sddlite_exp2_4gpus_ov-2021.4.582
EXP_NAME=${tmp1##*/}  # sddlite_exp2_4gpus_ov-2021.4.582

if [ ${TEST_SET} == "bit" ]; then
  IMAGE_DIR=/mnt/disk3/data_for_linjiaojiao/datasets/BITVehicle/images
  META_file=/mnt/disk3/data_for_linjiaojiao/datasets/BITVehicle/test_meta.list
else
  IMAGE_DIR=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/images
  META_file=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/val_meta.list
fi

source /mnt/disk1/data_for_linjiaojiao/intel/openvino_2021.4.582/bin/setupvars.sh
export PYTHONPATH=${ROOT}:${PYTHONPATH}

if [[ ! -d "results" ]]; then
  mkdir results
fi


if [[ ! -d "logs" ]]; then
  mkdir logs
fi


JOB_NAME=profile_ir_${EXP_NAME}_thr${THRESH}_${TEST_SET}
OUTPUT_DIR=results/${JOB_NAME}
echo 'start: ' ${JOB_NAME}
echo output_dir: ${OUTPUT_DIR}

python -u ${ROOT}/tools/openvino/run_vehicle_detection.py \
    --ir_xml ${IR_MODEL_FILE} \
    -m ${METHOD} \
    --images_dir ${IMAGE_DIR} \
    --list_file ${META_file} \
    -o ${OUTPUT_DIR} \
    -v 0 \
    --thresh ${THRESH} \
    2>&1 | tee logs/${JOB_NAME}.log

PRED_FILE=$(ls ${OUTPUT_DIR}/*.txt)
python -u ${ROOT}/tools/analysis_tools/eval_hybrid_list.py \
    --pred_file ${PRED_FILE} \
    --meta_file ${META_file} \
    --num_classes 1 \
    --thresh ${THRESH} \
    -v ${VIS_RATIO} \
    -o ${OUTPUT_DIR} \
    --images_dir ${IMAGE_DIR} \
    2>&1 | tee -a logs/${JOB_NAME}.log
