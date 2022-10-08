# Vehicle Detection

The repository is for training vehicle detection models. Based on mmdetection framework.

## Training models

Training scripts are located under `experiments/*/*/train_dist.sh`. To control the number of GPUs, use `export CUDA_VISIBLE_DEVICES` trick. 
Argments are specified using a config yaml file located in `configs/*`

## Evaluating models

Evaluation script is located under `experiments/*/*/test.sh`.
For convenience, config file used in testing stage will load from training-output-dir, for example, one experiment has been trained and saved in dir: output/base_exp0, test.sh try to load config file in `output/base_exp0/*.yaml`

## Exporting models to onnx and then to OpenVINO IR's

Check `experiments/toOpenvino/convert.sh` for the script.
Noted: we use OpenVINO op: DetectionOutput to replace the original post-processing, including bboxe decoder and NMS. See `--output_format openvino_op` in convert script.

[1]After successfully converting, we can evaluate IR models using `experiments/toOpenvino/run_ir.sh` to check accuracy and recall for the resulted models. 

[2]If you want to evaluate images without annotation, then use `experiments/toOpenvino/run_ir_on_images.sh` to do inference by specifying image folder. Use `-v {VIS_RATIO}` to save images with detectionresults.

## Benchmark

We use OpenVINO model: [vehicle-license-plate-detection-barrier-0106](https://docs.openvino.ai/2019_R1/_vehicle_license_plate_detection_barrier_0106_description_vehicle_license_plate_detection_barrier_0106.html) as baseline here.

### Model entry: Mobilenet-v2, SSD-based vehicle detection
And we also released a new model: Mobilenet-v2 SSD-based using this training repo.

The training and testing scripts can be found in `experiments/SSD/bit_vehicle`
training scripts: `experiments/SSD/bit_vehicle/train_dist.sh`
config file: `configs/ssd/ssdlite_mv2-ov-anchors_bit_vehicle_bgr.py`
dataset: BIT-Vehicle + UA-DETRAC-fps5


**Results**
 |model|dataset|data-type|recall@bit|precision@bit|recall@ua|precision@ua|
 |------|----|----|----|----|----|----|
 |vehicle-license-plate-detection-barrier-0106|unknown|IR-FP16|0.992|0.988|0.07|0.054|
 |vehicle-license-plate-detection-barrier-0106|unknown|IR-INT8|0.992|0.988|0.07|0.054|0.521|
 |release 0.0.1|BIT-Vehicle ; UA-DETRAC|Pytorch|0.992|0.991|0.742|0.625|0.808|
 |release 0.0.1|BIT-Vehicle ; UA-DETRAC|IR-FP16|0.992|0.991|0.72|0.615|0.803|
 |release 0.0.1|BIT-Vehicle ; UA-DETRAC|IR-INT8|0.992|0.991|0.713|0.614|0.8025|