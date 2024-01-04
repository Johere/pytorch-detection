#/bin/bash
ROOT=../../..
export PYTHONPATH=${ROOT}:${PYTHONPATH}

RESULTS_FILE=${1-"results/test_baseline_yolox_s_b64_coco6_2gpus/results.bbox.txt"}
RESULTS_DIR=${RESULTS_FILE%/*}  # output/test_baseline_yolox_s_b64_coco6_2gpus
EXP_NAME=${RESULTS_DIR##*/}  # test_baseline_yolox_s_b64_coco6_2gpus
THRESH=${2-0.5}
VIS_RATIO=${3-0.1}

IMAGE_DIR=/mnt/disk1/data_for_linjiaojiao/datasets/Radical/images
META_file=/mnt/disk1/data_for_linjiaojiao/datasets/Radical/images/radical_50m_collections_ford_edge_list_fps30.txt

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
    --num_classes 6 \
    --thresh ${THRESH} \
    -v ${VIS_RATIO} \
    -o ${OUTPUT_DIR} \
    --images_dir ${IMAGE_DIR} \
    2>&1 | tee -a logs/${JOB_NAME}.log
