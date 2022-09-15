import os
import argparse
import numpy as np
import cv2 as cv

# from openvino.tools.pot import DataLoader
# from openvino.tools.pot import IEEngine
# from openvino.tools.pot import load_model, save_model
# from openvino.tools.pot import compress_model_weights
# from openvino.tools.pot import create_pipeline

from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.pipeline.initializer import create_pipeline


parser = argparse.ArgumentParser(description='openvino quantization tool',
                                 allow_abbrev=False)
parser.add_argument(
    '--ir_xml',
    help='Path to detection IR model dir, /path/to/xml_file',
    default='/opt/models/vehicle-license-plate-detection-barrier-0106/FP16-INT8/'
            'vehicle-license-plate-detection-barrier-0106.xml'
)
parser.add_argument('--input_size', default=320, type=int, help='input size')
parser.add_argument('-m', '--quant_algorithm', default="default", choices=["default"], help='quantization algorithm')
parser.add_argument('--images_dir', type=str,
                    default='/home/linjiaojiao/remote_mount/datasets/BITVehicle/cropped_images',
                    help='image dir for quantization benchmark')
parser.add_argument('--list_file', type=str,
                    default=None,
                    # default='/home/linjiaojiao/remote_mount/datasets/UA_DETRAC_fps5/val_meta.list',
                    help='image list_file for quantization benchmark, default as None.')
args = parser.parse_args()


# Model config specifies the model name and paths to model .xml and .bin file
# model_config = {
#     "model_name": "model",
#     "model": path_to_xml,
#     "weights": path_to_bin,
# }

# Engine config
engine_config = {"device": "CPU"}

algorithms = {
    "default":
        {
            "name": "DefaultQuantization",
            "params": {
                "target_device": "ANY",
                "stat_subset_size": 300
            },
        }
}


class ImageLoader(DataLoader):
    """ Loads images from a folder """
    def __init__(self, dataset_path, input_size=320):
        # Use OpenCV to gather image files
        # Collect names of image files
        self._files = []
        all_files_in_dir = os.listdir(dataset_path)
        for name in all_files_in_dir:
            file = os.path.join(dataset_path, name)
            if cv.haveImageReader(file):
                self._files.append(file)

        # Define shape of the model
        self._shape = (input_size,input_size)

    def __len__(self):
        """ Returns the length of the dataset """
        return len(self._files)

    def __getitem__(self, index):
        """ Returns image data by index in the NCHW layout
        Note: model-specific preprocessing is omitted, consider adding it here
        """
        if index >= len(self):
            raise IndexError("Index out of dataset size")

        image = cv.imread(self._files[index]) # read image with OpenCV
        image = cv.resize(image, self._shape) # resize to a target input size
        image = np.expand_dims(image, 0)  # add batch dimension
        image = image.transpose(0, 3, 1, 2)  # convert to NCHW layout
        return image, None   # annotation is set to None


def quant_i8():
    print("start quantization: ", args.ir_xml)

    ir_dir = os.path.dirname(args.ir_xml)
    prec_dirname = os.path.basename(ir_dir)     # FP16
    output_dir = os.path.join(ir_dir, prec_dirname + "-INT8", args.quant_algorithm)     # FP16-INT8

    ir_bin = args.ir_xml.replace(".xml", ".bin")
    model_name = "model_" + str(args.input_size)

    model_config = {
        "model_name": model_name,
        "model": args.ir_xml,
        "weights": ir_bin,
    }

    # Step 1: Implement and create user's data loader
    data_loader = ImageLoader(args.images_dir)

    # Step 2: Load model
    model = load_model(model_config=model_config)

    # Step 3: Initialize the engine for metric calculation and statistics collection.
    engine = IEEngine(config=engine_config, data_loader=data_loader)

    # Step 4: Create a pipeline of compression algorithms and run it.
    pipeline = create_pipeline(algorithms[args.quant_algorithm], engine)
    compressed_model = pipeline.run(model=model)

    # Step 5 (Optional): Compress model weights to quantized precision
    #                     to reduce the size of the final .bin file.
    compress_model_weights(compressed_model)

    # Step 6: Save the compressed model to the desired path.
    # Set save_path to the directory where the model should be saved
    compressed_model_paths = save_model(
        model=compressed_model,
        save_path=output_dir,
        model_name=model_name,
    )
    compressed_model_path = compressed_model_paths[0]["model"]
    print("The quantized model is stored at", compressed_model_path)


if __name__ == '__main__':
    quant_i8()
    print("Done.")
