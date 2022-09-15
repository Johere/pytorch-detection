import shutil
import cv2
import os
import argparse
import glob
import time
import numpy as np

# source ~/intel/openvino_2021.4.582/bin/setupvars.sh
from tools.openvino.openvino_helper import get_IE_output

parser = argparse.ArgumentParser(description='run vehicle detection [need IR model]',
                                 allow_abbrev=False)
parser.add_argument(
    '--ir_xml',
    help='Path to detection IR model dir, /path/to/xml_file',
    default='/opt/models/vehicle-license-plate-detection-barrier-0106/FP16-INT8/'
            'vehicle-license-plate-detection-barrier-0106.xml'
)
parser.add_argument(
    '-m', '--method', default='barrier106', choices=['barrier106', 'ssdlite', 'ssdlite_barrier'], help='method for post-processing')
parser.add_argument('--images_dir', type=str,
                    default='/home/linjiaojiao/datasets/HCE_test/demo_image',
                    help='image dir for test')
parser.add_argument('--list_file', type=str,
                    default='/home/linjiaojiao/remote_mount/datasets/UA_DETRAC_fps5/val_meta.list',
                    help='image list_file for test')
parser.add_argument('-o', '--output_dir', type=str, default='./results/IR-models/vehicle_detector',
                    help='Path to dump predict results')
parser.add_argument('-v', '--vis_ratio', type=float, default=1, help='visualization ratio')
parser.add_argument('--thresh', type=float, default=-1, help='confidence threshold')
args = parser.parse_args()

"""
example:
python run_vehicle_detection.py \
    --ir_xml /opt/models/vehicle-license-plate-detection-barrier-0106/FP16-INT8/vehicle-license-plate-detection-barrier-0106.xml \
    -m barrier106 \
    --images_dir /home/linjiaojiao/remote_mount/datasets/BITVehicle/images \
    --list_file /home/linjiaojiao/remote_mount/datasets/BITVehicle/test_meta.list \
    -o ./results/IR-models/vehicle_detector \
    -v 1
"""

det2label = {
    0: 'background',
    1: 'vehicle',
    2: 'plate'
}

walk_cnt = 0
file_list = []


def walk_dir(path, key=None, filtered=None):
    """this is a statement"""
    global walk_cnt, file_list
    parents = os.listdir(path)
    for parent in parents:
        child = os.path.join(path, parent)
        # print(child)
        if os.path.isdir(child):
            walk_dir(child, key, filtered)
        else:
            skip = False
            if filtered is not None:
                for f in filtered:
                    if f in child:
                        skip = True
                        break
            if skip: continue

            if key is not None:
                skip = True
                if not isinstance(key, list):
                    key = [key]
                for k in key:
                    if k in child:
                        skip = False
                        break
            if skip: continue

            file_list.append(child)
            walk_cnt += 1
            if walk_cnt % 100 == 0:
                print("walk dir: %d" % walk_cnt)


def get_image(path, dst_size, color='bgr', flag='IR'):
    """
    :param color: bgr or rgb
    :param dst_size:
    :param path:
    :return:
    """
    if flag == 'IR':
        image = cv2.imread(path)
        return transform_fn(image, dst_size, color=color)
    else:
        raise ValueError('unknow flag:{}'.format(flag))


def transform_fn(image, dst_size, color='bgr'):
    if color is 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dst_size)
    x = image.transpose(2, 0, 1)
    return x


def glob_one_files(pattern):
    files = glob.glob(pattern)
    if len(files) == 1:
        return files[0]
    else:
        if len(files) == 0:
            print('IR model file not exists in {}'.format(pattern))
        else:
            print('multiple IR model files are found: {}'.format(files))
        raise ValueError


def inference_ssd_barrier_0106(image_files, input_size=(300, 300)):
    """
    vehicle-license-plate-detection-barrier-0106
    300x300. BGR

    IR dir: /opt/models/vehicle-license-plate-detection-barrier-0106/FP16-INT8/vehicle-license-plate-detection-barrier-0106.xml
    """
    color_space = 'bgr'
    # model_dir = '/opt/models/vehicle-license-plate-detection-barrier-0106/FP16-INT8/'
    model_dir = os.path.dirname(args.ir_xml)

    if args.vis_ratio > 0:
        vis_dir = os.path.join(args.output_dir, 'visualize')
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    det_vehicle_file = os.path.join(args.output_dir, 'vehicle_det_results.txt')
    fout = open(det_vehicle_file, 'w')
    fout.write('# path left top right bottom category-id score\n')

    executable_model = None
    net = None
    global_start = time.time()
    for idx, image_path in enumerate(image_files):
        end = time.time()

        image = cv2.imread(image_path)
        h, w, c = image.shape

        rel_image_path = image_path.replace(args.images_dir, '')
        while rel_image_path[0] == '/':
            rel_image_path = rel_image_path[1:]

        # detection
        ie_input = get_image(image_path, dst_size=input_size, color=color_space)

        executable_model, net, output = \
            get_IE_output(model_dir, ie_input, executable_model=executable_model, net=net)
        predictions = output['DetectionOutput_']

        cur_results = []
        # det_output: [1, 1, N, 7],  [image_id, label, conf, x_min, y_min, x_max, y_max]
        for obj in predictions[0][0]:
            # import pdb; pdb.set_trace()
            # image_id, cls, score, x_min, y_min, x_max, y_max = obj
            category_id = int(obj[1]) - 1
            score = obj[2]
            if args.thresh > 0 and score < args.thresh:
                continue
            label = det2label[int(obj[1])]
            try:
                x_min = int(w * obj[3])
                y_min = int(h * obj[4])
                x_max = int(w * obj[5])
                y_max = int(h * obj[6])
            except:
                import pdb; pdb.set_trace()

            # path score left top right down
            if label == 'vehicle':
                # path left top right bottom category-id score
                fout.write(
                    '{} {} {} {} {} {} {}\n'.format(rel_image_path, x_min, y_min, x_max, y_max, category_id, score))
            else:
                continue
                # raise ValueError('Unknown label:{}'.format(label))
            cur_results.append([label, score, x_min, y_min, x_max, y_max])

        if np.random.rand() <= args.vis_ratio:
            for res in cur_results:
                label, score, x_min, y_min, x_max, y_max = res
                cv2.putText(image, '{}: {:.2f}'.format(label, score), (x_min, y_min),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (232, 35, 244), 2)
            save_to = os.path.join(vis_dir, image_path.replace('/', '_'))
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            cv2.imwrite(save_to, image)

        elapsed = time.time() - end
        if idx % 10 == 0:
            fout.flush()
            print('[{}/{}] time: {:.03f} s'.format(idx + 1, len(image_files), elapsed))
    print('[{}] done. total time:{:.03f} s'.format(len(image_files), time.time() - global_start))

    fout.close()
    print('predicts saved: {}'.format(det_vehicle_file))
    if args.vis_ratio > 0:
        print('result images saved: {}'.format(vis_dir))


def inference_ssdlite(image_files):
    """
    ssdlite_mv2
    320x320. BGR

    """
    input_size = (320, 320)
    color_space = 'bgr'
    model_dir = os.path.dirname(args.ir_xml)

    if args.vis_ratio > 0:
        vis_dir = os.path.join(args.output_dir, 'visualize')
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    det_vehicle_file = os.path.join(args.output_dir, 'vehicle_det_results.txt')
    fout = open(det_vehicle_file, 'w')
    fout.write('# path left top right bottom category-id score\n')

    executable_model = None
    net = None
    global_start = time.time()
    for idx, image_path in enumerate(image_files):
        end = time.time()

        image = cv2.imread(image_path)
        h, w, c = image.shape

        rel_image_path = image_path.replace(args.images_dir, '')
        while rel_image_path[0] == '/':
            rel_image_path = rel_image_path[1:]

        # detection
        ie_input = get_image(image_path, dst_size=input_size, color=color_space)

        executable_model, net, output = \
            get_IE_output(model_dir, ie_input, executable_model=executable_model, net=net)
        '''
        (Pdb) output.keys()
            dict_keys(['TopK_1115.0', 'TopK_1353.0', 'TopK_1591.0', 'TopK_1829.0', 'TopK_2149.0', 'TopK_639.0', 'TopK_877.0', 'dets', 'labels'])
        (Pdb) output['dets'].shape
            (1, 200, 5)
        (Pdb) output['labels'].shape
            (1, 200)

        '''
        pred_bboxes = output['dets'][0]
        pred_cls = output['labels'][0]
        # import pdb; pdb.set_trace()
        scale_h = h / input_size[0] * 1.0
        scale_w = w / input_size[1] * 1.0

        cur_results = []
        for bbox, category_id in zip(pred_bboxes, pred_cls):
            score = bbox[-1]
            if args.thresh > 0 and score < args.thresh:
                continue
            label = det2label[category_id + 1]
            x_min = int(scale_w * bbox[0])
            y_min = int(scale_h * bbox[1])
            x_max = int(scale_w * bbox[2])
            y_max = int(scale_h * bbox[3])

            # path score left top right down
            if label == 'vehicle':
                # path left top right bottom category-id score
                fout.write(
                    '{} {} {} {} {} {} {}\n'.format(rel_image_path, x_min, y_min, x_max, y_max, category_id, score))
            else:
                continue
                # raise ValueError('Unknown label:{}'.format(label))
            cur_results.append([label, score, x_min, y_min, x_max, y_max])

        if np.random.rand() <= args.vis_ratio:
            for res in cur_results:
                label, score, x_min, y_min, x_max, y_max = res
                cv2.putText(image, '{}: {:.2f}'.format(label, score), (x_min, y_min),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (232, 35, 244), 2)
            save_to = os.path.join(vis_dir, image_path.replace('/', '_'))
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            cv2.imwrite(save_to, image)

        elapsed = time.time() - end
        if idx % 10 == 0:
            fout.flush()
            print('[{}/{}] time: {:.03f} s'.format(idx + 1, len(image_files), elapsed))
    print('[{}] done. total time:{:.03f} s'.format(len(image_files), time.time() - global_start))

    fout.close()
    print('predicts saved: {}'.format(det_vehicle_file))
    if args.vis_ratio > 0:
        print('result images saved: {}'.format(vis_dir))


if __name__ == '__main__':
    if args.list_file:
        with open(args.list_file, 'r') as f:
            lines = f.readlines()
        file_list = []
        for ln in lines:
            ln = ln.strip()
            if ln.startswith('#'):
                print('start reading list_file: {}'.format(ln))
                continue
            path = os.path.join(args.images_dir, ln.split()[0])
            file_list.append(path)
        file_list = list(set(file_list))
    else:
        walk_dir(args.images_dir, key=['.png', '.jpg'], filtered=['_mask.', 'visualize'])
    image_files = file_list
    print('{} images to be inferenced.'.format(len(image_files)))

    if args.method == 'barrier106':
        inference_ssd_barrier_0106(image_files)
    elif args.method == 'ssdlite':
        inference_ssdlite(image_files)
    elif args.method == 'ssdlite_barrier':
        inference_ssd_barrier_0106(image_files, input_size=(320, 320))
    else:
        raise ValueError("unknown post-process method: {}".format(args.method))
