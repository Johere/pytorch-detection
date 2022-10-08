import argparse
import os
import shutil

import cv2
from collections import defaultdict, OrderedDict
import numpy as np
from multiprocessing import Pool

from mmdet.core.evaluation import eval_map, eval_recalls
from mmdet.core.evaluation.mean_ap import get_cls_results, tpfp_default


parser = argparse.ArgumentParser(description='eval detection mAP offline, using hybrid-style list file',
                                 allow_abbrev=False)
parser.add_argument('-p', '--pred_file', type=str, required=True, help='Path to pred_file')
parser.add_argument('-m', '--meta_file', type=str, help='Path to meta_file for ground-truth (optional)')
parser.add_argument('--num_classes', type=int, default=1, help='num classes, default=1')
parser.add_argument('--thresh', type=float, default=-1, help='score threshold, default=-1, not filtering')
parser.add_argument('-o', '--output_dir', type=str, default='./results/detection_visualize',
                    help='Path to dump visualization results')
parser.add_argument('-v', '--vis_ratio', type=float, default=0, help='visualization ratio')
parser.add_argument('--images_dir', type=str,
                    default='/mnt/disk1/data_for_linjiaojiao/datasets/UA_DETRAC_fps5/images', help='image dir for dataset')
args = parser.parse_args()

tp_color_map = {
    0: (0, 0, 255),  # correct obj
    1: (0, 255, 0)   # wrong obj
}
gt_color = (255, 0, 0)


def parse_meta(list_file):
    """
    # path img-height img-width left top right bottom category vehicle-type orientation trajectory-length
    MVI_40855/img00025.jpg -1   -1      883.75  113.75  960.5   193.25  -1  -1      -1      -1
    MVI_40855/img00025.jpg 540  960     484     208     581     289     1   car     300.96  521

    return: data_infos:
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
    """
    PARSE_INDEX = {
        'path': -1,
        'img-height': -1,
        'img-width': -1,
        'left': -1,
        'top': -1,
        'right': -1,
        'bottom': -1
    }
    PARSE_OPTIONAL_INDEX = {
        'category': -1
    }
    def _get_val(contents, name):
        if name in PARSE_OPTIONAL_INDEX.keys():
            if PARSE_OPTIONAL_INDEX[name] == -1:
                return None
            else:
                return contents[PARSE_OPTIONAL_INDEX[name]]
        elif name in PARSE_INDEX.keys():
            return contents[PARSE_INDEX[name]]
        else:
            raise ValueError('unknonw column name: {}'.format(name))

    def _set_index(first_line):
    
        # initialize
        for key in PARSE_INDEX.keys():
            PARSE_INDEX[key] = -1
        for key in PARSE_OPTIONAL_INDEX.keys():
            PARSE_OPTIONAL_INDEX[key] = -1

        first_line = first_line.strip().replace('#', '')
        tmp = first_line.split()
        for key_idx, key in enumerate(tmp):
            if key in PARSE_INDEX.keys():
                PARSE_INDEX[key] = key_idx
            elif key in PARSE_OPTIONAL_INDEX.keys():
                PARSE_OPTIONAL_INDEX[key] = key_idx
            else:
                print('unknown key in meta_file: {}, ignore it!'.format(key))
        for key, val in PARSE_INDEX.items():
            if val == -1:
                raise ValueError('column not exist: {}'.format(key))
        print('================== PARSE_INDEX ====================')
        print(PARSE_INDEX)
        print('================== PARSE_OPTIONAL_INDEX ====================')
        print(PARSE_OPTIONAL_INDEX)

    print('------------------------------------------------------------')
    print('start parsing meta files: {}'.format(list_file))
    data_infos = []
    with open(list_file, 'r') as f:
        lines = f.readlines()

    ann_dict = defaultdict(list)
    img_info_dict = defaultdict(dict)
    for ln in lines:
        if ln.startswith("#"):
            _set_index(ln)
            continue
        ln = ln.strip()
        contents = ln.split()

        filename = _get_val(contents, 'path')
        img_height = _get_val(contents, 'img-height')
        img_width = _get_val(contents, 'img-width')
        left = _get_val(contents, 'left')
        top = _get_val(contents, 'top')
        right = _get_val(contents, 'right')
        bottom = _get_val(contents, 'bottom')

        # optional parts
        category = _get_val(contents, 'category')
        if category is None:
            category = 1
        else:
            category = int(category)

        ann_dict[filename].append([left, top, right, bottom, category])
        img_info_dict[filename] = {
            "width": int(img_width),
            "height": int(img_height)
        }

    cnt_objects = 0
    for filename, ann_list in ann_dict.items():

        img_info = img_info_dict[filename]

        cur_data_info = {
            "filename": filename,
            "width": img_info['width'],
            "height": img_info['height']
        }

        gt_bboxes = []
        gt_labels = []
        for item in ann_list:
            left, top, right, bottom, category = item
            try:
                bbox = [left, top, right, bottom]
                bbox = [int(float(_)) for _ in bbox]
                category = int(category)
            except:
                import pdb; pdb.set_trace()
            if category == -1:
                # ignore
                continue
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(0)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels)
        cur_data_info.update({'ann': ann})
        cnt_objects += gt_labels.shape[0]

        data_infos.append(cur_data_info)
    print('------------------------------------------------------------')
    print('parsing meta files done. Got: {} imgs, {} bboxes.'.format(len(ann_dict.keys()), cnt_objects))
    return data_infos


def evaluate(results,
             annotations,
             metric='mAP',
             logger=None,
             proposal_nums=(100, 300, 1000),
             iou_thr=0.5,
             scale_ranges=None,
             ds_name=['vehicle']
             ):
    """Evaluate in VOC protocol. Modified version towards coco

    Args:
        results (list[list | tuple]): Testing results of the dataset.
        metric (str | list[str]): Metrics to be evaluated. Options are
            'mAP', 'recall'.
        logger (logging.Logger | str, optional): Logger used for printing
            related information during evaluation. Default: None.
        proposal_nums (Sequence[int]): Proposal number used for evaluating
            recalls, such as recall@100, recall@1000.
            Default: (100, 300, 1000).
        iou_thr (float | list[float]): IoU threshold. Default: 0.5.
        scale_ranges (list[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Default: None.

    Returns:
        dict[str, float]: AP/recall metrics.
    """
    if not isinstance(metric, str):
        assert len(metric) == 1
        metric = metric[0]
    allowed_metrics = ['mAP', 'recall']
    if metric not in allowed_metrics:
        raise KeyError(f'metric {metric} is not supported')
    eval_results = OrderedDict()
    iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
    if metric == 'mAP':
        assert isinstance(iou_thrs, list)
        mean_aps = []
        for iou_thr in iou_thrs:
            print(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
            # Follow the official implementation,
            # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
            # we should use the legacy coordinate system in mmdet 1.x,
            # which means w, h should be computed as 'x2 - x1 + 1` and
            # `y2 - y1 + 1`
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr,
                dataset=ds_name,
                logger=logger,
                use_legacy_coordinate=True)
            mean_aps.append(mean_ap)
            eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
        eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        eval_results.move_to_end('mAP', last=False)
    elif metric == 'recall':
        gt_bboxes = [ann['bboxes'] for ann in annotations]
        recalls = eval_recalls(
            gt_bboxes,
            results,
            proposal_nums,
            iou_thrs,
            logger=logger,
            use_legacy_coordinate=True)
        for i, num in enumerate(proposal_nums):
            for j, iou_thr in enumerate(iou_thrs):
                eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
        if recalls.shape[1] > 1:
            ar = recalls.mean(axis=1)
            for i, num in enumerate(proposal_nums):
                eval_results[f'AR@{num}'] = ar[i]
    return eval_results


def get_tpfp(det_results,
             annotations,
             target_label=0,
             iou_thr=0.5,
             tpfp_fn=None,
             nproc=4):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        target_label(int): class_id
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_default` is used as default
            unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
            case). If it is given as a function, then this function is used
            to evaluate tp & fp. Default None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tp: numpy array of shape (1, num_dets)
        fp: numpy array of shape (1, num_dets)
    """
    assert len(det_results) == len(annotations)
    num_imgs = len(det_results)
    num_classes = len(det_results[0])  # positive class num
    assert target_label < num_classes, 'invalid target label: {}, num_classes = {}'.format(target_label, num_classes)

    pool = Pool(nproc)

    # get gt and det bboxes of this class
    cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
        det_results, annotations, target_label)
    # choose proper function according to datasets to compute tp and fp
    if tpfp_fn is None:
        tpfp_fn = tpfp_default
    if not callable(tpfp_fn):
        raise ValueError(
            f'tpfp_fn has to be a function or None, but got {tpfp_fn}')
    args = []
    # compute tp and fp for each image with multiple processes
    tpfp = pool.starmap(
        tpfp_fn,
        zip(cls_dets, cls_gts, cls_gts_ignore,
            [iou_thr for _ in range(num_imgs)], *args))
    tp, fp = tuple(zip(*tpfp))
    # calculate gt number of each scale
    # ignored gts or gts beyond the specific scale are not counted
    num_gts = np.zeros(1, dtype=int)
    for j, bbox in enumerate(cls_gts):
        num_gts[0] += bbox.shape[0]

    return tp, cls_dets, cls_gts


def visualize(tp, cls_dets, cls_gts, img_list, iou_thr=0.5):
    """visualize det results

        Args:
            img_list list(str): path
            cls_dets list(ndarray): Detected bboxes of target_label for each image, of shape (num_dets_per_img, 5)
            cls_gts list(ndarray): GT bboxes of target_label for each image, of shape (num_gts_per_img, 5)
            tp: list(ndarray): for each image: shape (num_dets_per_img, 1)
            iou_thr: 0.5

        """
    num_imgs = len(cls_dets)
    assert len(cls_gts) == num_imgs, 'mismatch: {} vs. {}'.format(len(cls_gts), num_imgs)
    assert len(tp) == num_imgs, 'mismatch: {} vs. {}'.format(len(tp), num_imgs)

    tmp = args.pred_file.split('/')
    prefix = tmp[-2]
    vis_cnt = 0
    vis_dir = os.path.join(args.output_dir, '{}_thresh{}'.format(prefix, str(args.thresh)))
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    for img_ix in range(num_imgs):
        filename = img_list[img_ix]
        img_path = os.path.join(args.images_dir, filename)
        if np.random.rand() > args.vis_ratio:
            continue
        assert os.path.exists(img_path), 'file not exists: {}'.format(img_path)
        image = cv2.imread(img_path)
        num_dets = cls_dets[img_ix].shape[0]
        det_bboxes = cls_dets[img_ix][:, :4]
        scores = cls_dets[img_ix][:, -1]
        tp_flag = tp[img_ix].reshape(num_dets)
        for det_ix in range(num_dets):
            score = scores[det_ix]
            if args.thresh > 0 and score < args.thresh:
                continue
            det_bbox = det_bboxes[det_ix]
            x_min = int(det_bbox[0])
            y_min = int(det_bbox[1])
            x_max = int(det_bbox[2])
            y_max = int(det_bbox[3])

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), tp_color_map[tp_flag[det_ix]], 2)
            cv2.putText(image, '{:.2f}'.format(score), (x_min, y_min),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, tp_color_map[tp_flag[det_ix]], thickness=2)

        num_gts = cls_gts[img_ix].shape[0]
        for gt_ix in range(num_gts):
            gt_bbox = cls_gts[img_ix][gt_ix]
            x_min = int(gt_bbox[0])
            y_min = int(gt_bbox[1])
            x_max = int(gt_bbox[2])
            y_max = int(gt_bbox[3])

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), gt_color, 1)

        save_to = os.path.join(vis_dir, filename.replace('/', '_'))
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        cv2.imwrite(save_to, image)
        vis_cnt += 1
        print('[{}/{}] visualized image saved: {}'.format(img_ix + 1, num_imgs, save_to))
    print('visualization done. {} images are saved.'.format(vis_cnt))
    print('visualization output dir:{}'.format(vis_dir))


if __name__ == '__main__':

    """
    hybrid-style pred_file
    # path left top right bottom category-id score
    
    > to form results:
    results (list[list]): [[cls1_det, cls2_det, ...], ...].
        The outer list indicates images, and the inner list indicates
        per-class detected bboxes.
    each-det is list[np.ndarray]:
        [np.array([left, top, right, bottom, score], ...)]
    """
    with open(args.pred_file, 'r') as f:
        lines = f.readlines()

    results_dict = defaultdict(list)
    for ln in lines:
        if ln.startswith('#'):
            continue
        tmp = ln.strip().split()
        path = tmp[0]
        left = float(tmp[1])
        top = float(tmp[2])
        right = float(tmp[3])
        bottom = float(tmp[4])
        label = int(tmp[5])
        assert label < args.num_classes, 'invalid label: {}, given num_classes = {}'.format(label, args.num_classes)
        score = float(tmp[6])
        results_dict[path].append([left, top, right, bottom, score])
    print('success to parse prediction file: {}, {} images'.format(args.pred_file, len(results_dict.keys())))

    """
    hybrid-style annotation meta_file
    # path img_height, img_width left top right bottom type
    
    > to form annotations:
    annotations (list[dict]): Ground truth annotations where each item of
        the list indicates an image. Keys of annotations are:
        - `bboxes`: numpy array of shape (n, 4)
        - `labels`: numpy array of shape (n, )
        - `bboxes_ignore` (optional): numpy array of shape (k, 4)
        - `labels_ignore` (optional): numpy array of shape (k, )
    for examples:
    [
        {
                'bboxes': < np.ndarray > (n, 4) in (x1, y1, x2, y2) order.
                'labels': < np.ndarray > (n,),
                'bboxes_ignore': < np.ndarray > (k, 4), (optional field)
                'labels_ignore': < np.ndarray > (k, 4)(optional field)
        },
        ...
    ]
    """
    data_infos = parse_meta(args.meta_file)
    annotations_dict = dict()
    for item in data_infos:
        filename = item['filename']
        annotations_dict[filename] = item['ann']
    print('success to parse meta_file: {}, {} images'.format(args.meta_file, len(data_infos)))

    results, annotations = [], []
    img_list = []
    for path in results_dict.keys():
        results.append([np.array(results_dict[path])])
        annotations.append(annotations_dict[path])
        img_list.append(path)

    print('start profiling...')
    tp, cls_dets, cls_gts = get_tpfp(results, annotations)
    if args.vis_ratio:
        visualize(tp, cls_dets, cls_gts, img_list)

    print('start evaluation...')
    eval_results = evaluate(results, annotations)
    print(eval_results)

    print('Done.')
