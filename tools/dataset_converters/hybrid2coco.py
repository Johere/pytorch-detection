# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from collections import defaultdict
import numpy as np
import mmcv
from PIL import Image
"""
> overall:
{
    "info": info,
    "licenses": [license],
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}
> details:
    info{
        "year": int,
        "version": str,
        "description": str,
        "contributor": str,
        "url": str,
        "date_created": datetime,
    }
    license{
        "id": int,
        "name": str,
        "url": str,
    } 
    image{
        "id": int,
        "width": int,
        "height": int,
        "file_name": str,
        "license": int,
        "flickr_url": str,
        "coco_url": str,
        "date_captured": datetime,
    }
    annotation{
        "id": int,    
        "image_id": int,
        "category_id": int,
        "segmentation": RLE or [polygon],
        "area": float,
        "bbox": [x,y,width,height],
        "iscrowd": 0 or 1,
    }
"""

CATEGORY_ID_OFFSET = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert hybrid-style dataset to coco format')
    parser.add_argument('--meta_file', type=str, help='meta file for hybrid-style dataset')
    parser.add_argument('--dataset_dir', type=str, default=None, help='dataset dir')
    parser.add_argument(
        '--classes', type=str, default=None, help='The text file name of storage class list')
    parser.add_argument(
        '-o', '--out',
        type=str,
        default=None,
        help='The output annotation json file name, The save dir is in the '
        'same directory as meta_file')
    args = parser.parse_args()
    return args


def collect_image_infos(meta_file):
    print('start parsing meta_file: {}'.format(meta_file))
    with open(meta_file, 'r') as f:
        lines = f.readlines()

    img_infos = []
    anno_infos = defaultdict(list)
    image_set = set()
    for ln in lines:
        ln = ln.strip()
        if ln.startswith('#'):
            print('start parsing: {}'.format(ln))
            continue

        contents = ln.split()[:7]
        filename, img_height, img_width, left, top, right, bottom = contents

        if filename not in image_set:
            img_info = {
                'filename': filename,
                'width': img_width,
                'height': img_height,
            }
            img_infos.append(img_info)
            image_set.add(filename)

        anno_infos[filename].append([int(left), int(top), int(right), int(bottom)])
    return img_infos, anno_infos


def xyxy2xywh(bbox):
    """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
    evaluation.

    Args:
        bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
            ``xyxy`` order.

    Returns:
        list[float]: The converted bounding boxes, in ``xywh`` order.
    """

    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]


def cvt_to_coco_json(img_infos, classes, anno_infos=None):
    image_id = 0
    anno_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    for category_id, name in enumerate(classes):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for img_dict in img_infos:
        file_name = img_dict['filename']
        assert file_name not in image_set

        '''
        image{
            "id": int,
            "width": int,
            "height": int,
            "file_name": str,
            "license": int,
            "flickr_url": str,
            "coco_url": str,
            "date_captured": datetime,
        }
        '''
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(img_dict['height'])
        image_item['width'] = int(img_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        if anno_infos is not None:
            '''  
            annotation{
                "id": int,    
                "image_id": int,
                "category_id": int,
                "segmentation": RLE or [polygon],
                "area": float,
                "bbox": [x,y,width,height],
                "iscrowd": 0 or 1,
            }
            '''
            annotations = anno_infos[file_name]
            for anno in annotations:
                if len(anno) == 4:  # bbox
                    left, top, right, bottom = anno
                    category_id = CATEGORY_ID_OFFSET
                elif len(anno) == 5:
                    left, top, right, bottom, category_id = anno
                else:
                    raise ValueError('invalid annotation:{}'.format(anno))

                anno_item = dict()
                anno_item['id'] = int(anno_id)
                anno_item['image_id'] = int(image_id)
                anno_item['category_id'] = int(category_id)
                anno_item['segmentation'] = []
                anno_item['area'] = 0
                anno_item['bbox'] = xyxy2xywh(np.array([left, top, right, bottom]))
                anno_item['iscrowd'] = 0
                coco['annotations'].append(anno_item)
                anno_id += 1

        image_id += 1
    return coco


def main():
    args = parse_args()
    if args.out:
        assert args.out.endswith(
            'json'), 'The output file name must be json suffix'
    else:
        prefix, ext = os.path.splitext(os.path.basename(args.meta_file))
        args.out = prefix + '.json'

    # 1 load image list info
    # if args.dataset_dir:
    #     img_prefix = args.dataset_dir
    # else:
    #     img_prefix = os.path.dirname(args.meta_file)
    img_infos, anno_infos = collect_image_infos(args.meta_file)
    print('collecting {} images'.format(len(img_infos)))

    # 2 convert to coco format data
    if args.classes:
        raise NotImplemented
    else:
        classes = ['vehicle']
    coco_info = cvt_to_coco_json(img_infos, classes, anno_infos)

    # 3 dump
    save_dir = os.path.join(os.path.dirname(args.meta_file), 'coco_annotation')
    mmcv.mkdir_or_exist(save_dir)
    save_path = os.path.join(save_dir, args.out)
    mmcv.dump(coco_info, save_path)
    print(f'save json file: {save_path}')


if __name__ == '__main__':
    main()
