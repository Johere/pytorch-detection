import argparse
import os
import shutil
from tqdm import tqdm
import cv2
from collections import defaultdict
import numpy as np
import copy

from mmdet.datasets.hybrid import HybridDataset


parser = argparse.ArgumentParser(description='crop vehicle rois to paste on background, using hybrid-style list file',
                                 allow_abbrev=False)
parser.add_argument('-s', '--src_list', type=str, required=True, help='Path to source roi list, require absolute path')
parser.add_argument('-m', '--meta_file', type=str,
                    default='/mnt/disk1/data_for_linjiaojiao/datasets/BITVehicle/train_meta.list',
                    help='Path to meta_file, using to make background images')
parser.add_argument('--images_dir', type=str,
                    default=None,
                    # default='/mnt/disk1/data_for_linjiaojiao/datasets/BITVehicle/images',
                    help='image dir for dataset')
parser.add_argument('--min_rois', type=int, default=1, help='minimum rois in one background')
parser.add_argument('--max_rois', type=int, default=3, help='maximum rois in one background')
parser.add_argument('-v', '--vis_cnt', type=int, default=0, help='visualize count of background image')
parser.add_argument('-o', '--output_dir', type=str, default='/mnt/disk3/data_for_linjiaojiao/datasets/augment_copy_and_paste',
                    help='Path to dump visualization results')
args = parser.parse_args()

'''
export PYTHONPATH=/mnt/disk1/data_for_linjiaojiao/projects/mmdetection:$PYTHONPATH
python augment_copy_paste.py -s /mnt/disk3/data_for_linjiaojiao/datasets/Cars-dataset/cropped_images_abs.list -o /mnt/disk3/data_for_linjiaojiao/datasets/augment_copy_and_paste_bit
python augment_copy_paste.py -s /mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/cropped_images_abs.list -o /mnt/disk3/data_for_linjiaojiao/datasets/augment_copy_and_paste_bit
python augment_copy_paste.py -s /mnt/disk3/data_for_linjiaojiao/datasets/VMMRdb/images_abs.list -o /mnt/disk3/data_for_linjiaojiao/datasets/augment_copy_and_paste_bit

python augment_copy_paste.py -s /mnt/disk3/data_for_linjiaojiao/datasets/BoxCars21k/cropped_images_abs.list \
-m /mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/train_meta.list -o /mnt/disk3/data_for_linjiaojiao/datasets/augment_copy_and_paste_uadetrac-v2
python augment_copy_paste.py -s /mnt/disk3/data_for_linjiaojiao/datasets/Cars-dataset/images_abs.list \
-m /mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/train_meta.list -o /mnt/disk3/data_for_linjiaojiao/datasets/augment_copy_and_paste_uadetrac-v2
python augment_copy_paste.py -s /mnt/disk3/data_for_linjiaojiao/datasets/VMMRdb/images_abs.list \
-m /mnt/disk3/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/train_meta.list -o /mnt/disk3/data_for_linjiaojiao/datasets/augment_copy_and_paste_uadetrac-v2
'''


class MiniHybridDataset(HybridDataset):
    def __init__(self, list_file):
        self.data_infos = self.load_annotations(list_file)


def parse_rois(list_file):
    """
    # absolute_path
    000001_0.jpg
    """
    print('start parsing source roi list file: {}'.format(list_file))

    with open(list_file, 'r') as f:
        lines = f.readlines()

    roi_list = []
    for ln in lines:
        path = ln.strip().split()[0]
        roi_list.append(path)

    return roi_list


def generate_background(list_file, output_dir=None):
    """
    data_infos:
    [{
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann':
            {
                    'bboxes': < np.ndarray > (n, 4) in (x1, y1, x2, y2) order.
                    'labels': < np.ndarray > (n,),
                    'bboxes_ignore': < np.ndarray > (k, 4), (optional field)
                    'labels_ignore': < np.ndarray > (k, 4)(optional field)
            }
    },
    ...]
    return:
    bg_list:
        [filename] : {
            'image': numpay.ndarray
            'hw_statistic': [avg_box_h, avg_box_w]
        }
    """
    print('start to generate background images: {}'.format(list_file))
    hybrid_mini = MiniHybridDataset(list_file)
    data_infos = hybrid_mini.data_infos

    vis_ratio = 0
    if args.vis_cnt > 0:
        vis_ratio = args.vis_cnt * 1.0 / len(data_infos)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    if args.images_dir:
        images_dir = args.images_dir
    else:
        images_dir = os.path.join(os.path.dirname(list_file), 'images')

    bg_list = dict()
    for item in tqdm(data_infos):
        filename = item['filename']
        _data = dict()
        path = os.path.join(images_dir, filename)
        assert os.path.exists(path), 'file not exist: {}'.format(path)
        image = cv2.imread(path)

        bboxes_ignore = item['ann']['bboxes_ignore']
        for ibox in bboxes_ignore:
            x1, y1, x2, y2 = [int(_) for _ in ibox]
            crop = image[y1: y2, x1: x2, :]
            mean = np.mean(crop)
            image[y1: y2, x1: x2, :] = mean

        bboxes = item['ann']['bboxes']
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(_) for _ in bbox]
            crop = image[y1: y2, x1: x2, :]
            mean = np.mean(crop)
            image[y1: y2, x1: x2, :] = mean
        _data['image'] = image

        if np.random.rand() <= vis_ratio:
            save_to = os.path.join(output_dir, filename.replace('/', '_'))
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            cv2.imwrite(save_to, image)
            print('background file saved: {}'.format(save_to))

        if bboxes.shape[0] > 0:
            ws = bboxes[:, 2] - bboxes[:, 0]
            hs = bboxes[:, 3] - bboxes[:, 1]
            avg_box_hw = [np.mean(ws), np.mean(hs)]
        else:
            width = item['width']
            height = item['height']
            avg_box_hw = [height / 10.0, width / 10.0]
        _data['hw_statistic'] = [int(_) for _ in avg_box_hw]
        # #debug
        # if len(bg_list.keys()) > 10:
        #     break

        bg_list[filename] = _data
    return bg_list


def paste2bg(roi_list, bg_images_dict, output_dir=None):
    """
    bg_images_dict:
        [filename] : {
            'image': numpay.ndarray
            'hw_statistic': [avg_box_h, avg_box_w]
        }
    """
    print('start to generate synthesis images, rois: {} backgrounds: {}, with min_rois = {}, max_rois = {}'.
          format(len(roi_list), len(bg_images_dict.keys()), args.min_rois, args.max_rois))

    bg_filename_list = list(bg_images_dict.keys())

    repeat_times = len(roi_list) // args.max_rois // len(bg_filename_list)
    print('roi to background ratio: {}'.format(repeat_times))
    if repeat_times > 1:
        repeat_times = min(repeat_times, 5)
    else:
        repeat_times = 1
    print('repeat background image list: {}'.format(repeat_times))

    output_images_dir = os.path.join(output_dir, 'images')
    output_meta_file = os.path.join(output_dir, 'meta.list')
    fout = open(output_meta_file, 'w')
    fout.write('# path img-height img-width left top right bottom\n')

    roi_indices = np.arange(len(roi_list))
    roi_flags = np.zeros(len(roi_indices))
    res_cnt = 0
    for repeat_ix in range(repeat_times):
        print('======================================================')
        print('repeat idx: {}'.format(repeat_ix))
        print('======================================================')
        for bg_filename in bg_filename_list:

            bg_img = copy.deepcopy(bg_images_dict[bg_filename]['image'])
            num_rois = np.random.randint(args.min_rois, args.max_rois + 1)
            filtered_indices = np.where(roi_flags == 0)[0]
            if len(filtered_indices) == 0:
                break
            indices = np.random.choice(filtered_indices, num_rois)

            prefix, ext = os.path.splitext(bg_filename)
            dst_filename = '{}_{}rois_{}{}'.format(prefix, num_rois, repeat_ix, ext)

            H, W, C = bg_img.shape
            start_x = 0
            start_y = 0

            box_avg_hw = bg_images_dict[bg_filename]['hw_statistic']
            for roi_idx in indices:

                roi_path = roi_list[roi_idx]
                assert os.path.exists(roi_path), 'file not exist: {}'.format(roi_path)
                roi_img = cv2.imread(roi_path)
                roi_h, roi_w, _ = roi_img.shape

                if 'UA_DETRAC' in args.meta_file:
                    box_scale = np.random.choice(np.arange(1.0, 2.0, 0.1))
                else:
                    box_scale = np.random.choice(np.arange(0.6, 1.2, 0.1))
                paste_w = int(box_avg_hw[1] * box_scale)  # use roi-width as anchor
                paste_h = int(roi_h / roi_w * paste_w)

                end_x = int(min(W * 9.0 / 10, W - paste_w))
                end_y = int(min(H * 9.0 / 10, H - paste_h))
                if end_x <= start_x:
                    print('invalid try, start_x: {}, end_x: {}'.format(start_x, end_x))
                    continue
                if end_y <= start_y:
                    print('invalid try, start_y: {}, end_y: {}'.format(start_y, end_y))
                    continue

                left = np.random.randint(start_x, end_x)
                top = np.random.randint(start_y, end_y)
                right = left + paste_w
                bottom = top + paste_h

                scaled_roi_img = cv2.resize(roi_img, (paste_w, paste_h))
                if np.random.rand() <= 0.5:
                    scaled_roi_img = scaled_roi_img[:, ::-1, :]
                bg_img[top: bottom, left: right, :] = scaled_roi_img
                meta_str = '{} {} {} {} {} {} {}\n'.format(dst_filename, H, W, left, top, right, bottom)
                fout.write(meta_str)

                roi_flags[roi_idx] = 1

            save_to = os.path.join(output_images_dir, dst_filename)
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            cv2.imwrite(save_to, bg_img)
            # print('file saved: {}'.format(save_to))

            res_cnt += 1
            if res_cnt % 10 == 0:
                print('[{}/{}] background images processed. {}/{} rois are involved.'.
                      format(res_cnt, len(bg_filename_list) * repeat_times, np.sum(roi_flags), len(roi_flags)))
                fout.flush()
    print('{} background images processed. {}/{} rois are involved.'.format(res_cnt, np.sum(roi_flags), len(roi_flags)))
    fout.close()
    print('file saved: {}'.format(output_meta_file))


if __name__ == '__main__':
    np.random.seed(131)

    tmp = args.meta_file.split('/')[-2:]
    bg_prefix = '_'.join(tmp).replace('.', '_')
    bg_output_dir = os.path.join(args.output_dir, bg_prefix + '_background_vis')
    bg_images = generate_background(args.meta_file, output_dir=bg_output_dir)

    tmp = args.src_list.split('/')[-2:]
    prefix = '_'.join(tmp).replace('.', '_')
    aug_output_dir = os.path.join(args.output_dir, prefix)
    if os.path.exists(aug_output_dir):
        shutil.rmtree(aug_output_dir)
    os.makedirs(aug_output_dir)
    roi_list = parse_rois(args.src_list)
    paste2bg(roi_list, bg_images, output_dir=aug_output_dir)

    print('Done.')
