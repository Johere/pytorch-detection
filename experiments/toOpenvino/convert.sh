#/bin/bash
# openvino/model-optimizer
ROOT=../..
export PYTHONPATH=${ROOT}:${PYTHONPATH}
IR_DATA_TYPE=FP16  # FP16, FP32
OV_VERSION=2021.4.582
#OV_VERSION=2021.4.752

if [[ ! -d "output" ]]; then
  mkdir output
fi


if [[ ! -d "logs" ]]; then
  mkdir logs
fi

CHECKPOINT_FILE=${1-"../SSD/bit_vehicle/output/ssdlite_exp13.1_4gpus/latest.pth"}
MODEL_DIR=${CHECKPOINT_FILE%/*}
CONFIG_FILE=$(ls ${MODEL_DIR}/*.py)

INPUT_SHAPE=${2-"300"}

EXP_NAME=${MODEL_DIR##*/}
OUTPUT_DIR=output/${EXP_NAME}_input${INPUT_SHAPE}_openvino-op
mkdir -p ${OUTPUT_DIR}
OUTPUT_FILE=${OUTPUT_DIR}/${CHECKPOINT_FILE##*/}

echo checkpoint: ${CHECKPOINT_FILE}, config: ${CONFIG_FILE}, input_shape: ${INPUT_SHAPE}
echo ir_data_type: ${IR_DATA_TYPE} openvino version: ${OV_VERSION}
echo output_dir: ${OUTPUT_DIR}

echo '========== start converting' ${EXP_NAME} '=========='
toONNXModel(){
  export CUDA_VISIBLE_DEVICES='1'
  JOD_NAME=to_onnx_${EXP_NAME}
  echo ${JOD_NAME}
  python ${ROOT}/tools/deployment/pytorch2onnx.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --output-file ${OUTPUT_FILE}.onnx \
    --input-img ${ROOT}/tests/data/color.jpg \
    --shape ${INPUT_SHAPE} \
    --dynamic-export \
    --cfg-options \
      model.test_cfg.deploy_nms_pre=-1 \
    --show \
    --output_format openvino_op \
    2>&1 | tee logs/${JOD_NAME}.log
  echo ${JOD_NAME} 'done.'
#    --test-img ${TEST_IMAGE_PATH} \
#    --simplify \ # onnx will generate shape->gather->unsqueeze->reshape for op: reshape, using this to simplify it!!!
#    --opset-version ${OPSET_VERSION} \
#    --cfg-options ${CFG_OPTIONS}
#    --output_format single_output \
#    --output_format openvino_op \  # use openvino op: `DetectionOutput`
    # --verify \    # use custom op: DetectionOutput, so we don't verify the generated onnx model
}
simplifyONNX(){
  # input_model output_model [check_n] [args]
  mv ${OUTPUT_FILE}.onnx ${OUTPUT_FILE}-naive.onnx
  python -m onnxsim ${OUTPUT_FILE}-naive.onnx ${OUTPUT_FILE}.onnx --input-shape 1,3,${INPUT_SHAPE},${INPUT_SHAPE}
}
toMOModel(){
  export OPENVINO_PATH=/mnt/disk1/data_for_linjiaojiao/intel/openvino_${OV_VERSION}
  source ${OPENVINO_PATH}/bin/setupvars.sh
  export CUDA_VISIBLE_DEVICES=''
  ONNX_MODEL=${OUTPUT_FILE}.onnx
  MO_OUTPUT_DIR=${OUTPUT_DIR}/${IR_DATA_TYPE}
  JOD_NAME=to_mo-${EXP_NAME}_${IR_DATA_TYPE}
  echo ${JOD_NAME}
  python ${OPENVINO_PATH}/deployment_tools/model_optimizer/mo_onnx.py \
		--input_model ${ONNX_MODEL} \
		--mean_values '(103.530,116.280,123.675)' \
		--scale_values '(57.375, 57.12, 58.395)' \
		--data_type ${IR_DATA_TYPE} \
		--output_dir ${MO_OUTPUT_DIR} \
		--log_level=DEBUG \
    2>&1 | tee logs/${JOD_NAME}.log
  echo ${MO_OUTPUT_DIR} 'saved.'
  echo ${JOD_NAME} 'done.'
}
toONNXModel;
simplifyONNX;
toMOModel;
#   --reverse_input_channels   # to get RGB mode
#		--batch 1 \
#		--input_shape=[1,3,${INPUT_SHAPE},${INPUT_SHAPE}] \

#    --log_level=DEBUG \
#		--input "input" \
#		--input_shape=[1,3,320,320] \
#		--output=TopK_639:0,TopK_877:0,TopK_1115:0,TopK_1353:0,TopK_1591:0,TopK_1829:0,TopK_2149:0 \
    # --output=Concat_695,Concat_862 \
    # --output=Concat_689,Concat_856 \   # remove softmax
