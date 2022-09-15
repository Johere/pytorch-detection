#/bin/bash
# openvino/model-optimizer
ROOT=../..
export PYTHONPATH=${ROOT}:${PYTHONPATH}

CONFIG_FILE=${1-${ROOT}/configs/ssd/ssdlite_0.5mv2_bit_vehicle_bgr.py}

INPUT_SHAPE=${2-"300"}
mkdir -p ./logs

JOB_NAME=flops_${CONFIG_FILE##*/}

echo config: ${CONFIG_FILE}, input_shape: ${INPUT_SHAPE}

#python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
python ${ROOT}/tools/analysis_tools/get_flops.py \
  ${CONFIG_FILE} \
  --shape ${INPUT_SHAPE} \
  --size-divisor -1 \
    2>&1 | tee logs/${JOB_NAME}.log

echo 'Done.'
