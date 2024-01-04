#/bin/bash
# openvino/model-optimizer
ROOT=../../..
export PYTHONPATH=${ROOT}:${PYTHONPATH}
IR_DATA_TYPE=FP16  # FP16, FP32
OV_VERSION=2022.3.1

if [[ ! -d "output" ]]; then
  mkdir output
fi


if [[ ! -d "logs" ]]; then
  mkdir logs
fi

CHECKPOINT_FILE=${1-"output/baseline_yolox_s_b64_coco6_2gpus/latest.pth"}
MODEL_DIR=${CHECKPOINT_FILE%/*}
CONFIG_FILE=$(ls ${MODEL_DIR}/*.py)

INPUT_SHAPE=${2-"640"}

EXP_NAME=${MODEL_DIR##*/}
OUTPUT_DIR=output/${EXP_NAME}_input${INPUT_SHAPE}
mkdir -p ${OUTPUT_DIR}
OUTPUT_FILE=${OUTPUT_DIR}/${CHECKPOINT_FILE##*/}

echo checkpoint: ${CHECKPOINT_FILE}, config: ${CONFIG_FILE}, input_shape: ${INPUT_SHAPE}
echo ir_data_type: ${IR_DATA_TYPE} openvino version: ${OV_VERSION}
echo output_dir: ${OUTPUT_DIR}

echo '========== start converting' ${EXP_NAME} '=========='

toONNXModel(){

  # success at torch: 1.13.1

  export CUDA_VISIBLE_DEVICES='1'
  JOD_NAME=to_onnx_${EXP_NAME}
  echo ${JOD_NAME}
  python ${ROOT}/tools/deployment/to_onnx.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --output-file ${OUTPUT_FILE}.onnx \
    --input-img ${ROOT}/tests/data/color.jpg \
    --shape ${INPUT_SHAPE} \
    --cfg-options \
      model.test_cfg.deploy_nms_pre=-1 \
    --show \
    2>&1 | tee logs/${JOD_NAME}.log
  echo ${JOD_NAME} 'done.'
}
simplifyONNX(){
  # input_model output_model [check_n] [args]
  mv ${OUTPUT_FILE}.onnx ${OUTPUT_FILE}-naive.onnx
  python -m onnxsim ${OUTPUT_FILE}-naive.onnx ${OUTPUT_FILE}.onnx --input-shape 1,3,${INPUT_SHAPE},${INPUT_SHAPE}
}
toMOModel(){
  # data['img_metas']:
  #   DataContainer({'filename': '', 'ori_filename': '', 
  #     'ori_shape': (720, 1280, 3), 'img_shape': (640, 640, 3), 
  #     'pad_shape': (640, 640, 3), 
  #     'scale_factor': array([0.5      , 0.8888889, 0.5      , 0.8888889], dtype=float32), 
  #     'flip': False, 'flip_direction': None, 
  #     'img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 
  #     'std': array([1., 1., 1.], dtype=float32), 
  #     'to_rgb': False}}
  ONNX_MODEL=${OUTPUT_FILE}.onnx
  MO_OUTPUT_DIR=${OUTPUT_DIR}/${IR_DATA_TYPE}
  JOD_NAME=to_mo-${EXP_NAME}_${IR_DATA_TYPE}
  echo ${JOD_NAME}
  mo --input_model ${ONNX_MODEL} \
     --model_name yolox_s \
     --output outputs \
     --data_type ${IR_DATA_TYPE} \
     --output_dir ${MO_OUTPUT_DIR} \
     2>&1 | tee logs/${JOD_NAME}.log
# --reverse_input_channels \      # to get RGB mode
}
quantModel(){
  pot -q default --engine simplified \
  -m ${MO_OUTPUT_DIR}/yolox_s.xml -w ${MO_OUTPUT_DIR}/yolox_s.bin \
  --name yolox_s \
  --output-dir ${OUTPUT_DIR}/${IR_DATA_TYPE}-INT8 -d \
  --data-source /mnt/disk2/data_for_linjiaojiao/datasets/coco2017/COCO/val2017
}
toONNXModel;
simplifyONNX;
toMOModel;
quantModel;
