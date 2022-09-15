#/bin/bash
ROOT=../../..

RESULTS_FILE=${1-"output/test_bit-test_sddlite_exp4_4gpus/results.bbox.txt"}
RESULTS_DIR=${RESULTS_FILE%/*}  # output/test_bit-test_sddlite_exp4_4gpus
EXP_NAME=${RESULTS_DIR##*/}  # test_bit-test_sddlite_exp4_4gpus
THRESH=${2-0.5}
VIS_RATIO=${3-1}
TEST_SET=${4-"bit"}  # bit, uadetrac

if [ ${TEST_SET} == "bit" ]; then
  IMAGE_DIR=/mnt/disk3/data_for_linjiaojiao/datasets/BITVehicle/images
  META_file=/mnt/disk3/data_for_linjiaojiao/datasets/BITVehicle/test_meta.list
else
  IMAGE_DIR=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/images
  META_file=/mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/val_meta.list
fi

if [[ ! -d "logs" ]]; then
  mkdir logs
fi


JOB_NAME=visualize_${EXP_NAME}_thr${THRESH}_${TEST_SET}
OUTPUT_DIR=${RESULTS_DIR}/visualize
echo 'start: ' ${JOB_NAME}
echo output_dir: ${OUTPUT_DIR}

python -u ${ROOT}/tools/analysis_tools/eval_hybrid_list.py \
    --pred_file ${RESULTS_FILE} \
    --meta_file ${META_file} \
    --num_classes 1 \
    --thresh ${THRESH} \
    -v ${VIS_RATIO} \
    -o ${OUTPUT_DIR} \
    --images_dir ${IMAGE_DIR} \
    2>&1 | tee -a logs/${JOB_NAME}.log
