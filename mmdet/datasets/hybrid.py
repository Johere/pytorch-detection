"""
author: jiaojiao.lin@intel.com
"""
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_map, eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class HybridDataset(CustomDataset):
    """
    required:
    self.data_infos:
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

    key function: can inherit from CustomDataset
    [1] load_annotations(self, ann_file)
    [2] get_ann_info(self, idx)
    """
    CLASSES = ['vehicle']

    PALETTE = [(220, 20, 60)]

    '''
    !!! must starts with one comment line to specific each column !!!
        # path img-height img-width left top right bottom category ***
    '''
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

    def _get_val(self, contents, name):
        if name in self.PARSE_OPTIONAL_INDEX.keys():
            if self.PARSE_OPTIONAL_INDEX[name] == -1:
                return None
            else:
                return contents[self.PARSE_OPTIONAL_INDEX[name]]
        elif name in self.PARSE_INDEX.keys():
            return contents[self.PARSE_INDEX[name]]
        else:
            raise ValueError('unknonw column name: {}'.format(name))

    def _set_index(self, first_line):

        # initialize
        for key in self.PARSE_INDEX.keys():
            self.PARSE_INDEX[key] = -1
        for key in self.PARSE_OPTIONAL_INDEX.keys():
            self.PARSE_OPTIONAL_INDEX[key] = -1

        first_line = first_line.strip().replace('#', '')
        tmp = first_line.split()
        for key_idx, key in enumerate(tmp):
            if key in self.PARSE_INDEX.keys():
                self.PARSE_INDEX[key] = key_idx
            elif key in self.PARSE_OPTIONAL_INDEX.keys():
                self.PARSE_OPTIONAL_INDEX[key] = key_idx
            else:
                print('unknown key in meta_file: {}, ignore it!'.format(key))
        for key, val in self.PARSE_INDEX.items():
            if val == -1:
                raise ValueError('column not exist: {}'.format(key))
        # print('================== self.PARSE_INDEX ====================')
        # print(self.PARSE_INDEX)
        # print('================== self.PARSE_OPTIONAL_INDEX ====================')
        # print(self.PARSE_OPTIONAL_INDEX)

    def _parse_meta(self, list_file):
        """
        # path img-height img-width left top right bottom category vehicle-type orientation trajectory-length
        MVI_40855/img00005.jpg 540 960 481 205 581 286 1 car 225 521
        """
        print('------------------------------------------------------------')
        print('start parsing meta files: {}'.format(list_file))
        data_infos = []
        with open(list_file, 'r') as f:
            lines = f.readlines()

        self._set_index(lines[0])

        ann_dict = defaultdict(list)
        img_info_dict = defaultdict(dict)
        for ln in lines[1:]:
            if ln.startswith("#"):
                continue
            ln = ln.strip()

            contents = ln.split()

            filename = self._get_val(contents, 'path')
            img_height = self._get_val(contents, 'img-height')
            img_width = self._get_val(contents, 'img-width')
            left = self._get_val(contents, 'left')
            top = self._get_val(contents, 'top')
            right = self._get_val(contents, 'right')
            bottom = self._get_val(contents, 'bottom')

            # optional parts
            category = self._get_val(contents, 'category')
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

            gt_bboxes_ignore = []
            gt_bboxes = []
            gt_labels = []
            for item in ann_list:
                left, top, right, bottom, category = item
                bbox = [left, top, right, bottom]
                bbox = [int(float(_)) for _ in bbox]

                if category == -1:
                    # ignore bboxes
                    gt_bboxes_ignore.append(bbox)
                else:
                    gt_bboxes.append(bbox)
                    gt_labels.append(self.cat2label[category - 1])

            if gt_bboxes:
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                gt_labels = np.array(gt_labels, dtype=np.int64)
            else:
                gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.array([], dtype=np.int64)

            if gt_bboxes_ignore:
                gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
            else:
                gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

            ann = dict(
                bboxes=gt_bboxes,
                labels=gt_labels,
                bboxes_ignore=gt_bboxes_ignore
            )
            cur_data_info.update({'ann': ann})
            cnt_objects += gt_labels.shape[0]

            data_infos.append(cur_data_info)
        print('------------------------------------------------------------')
        print('parsing meta files done. Got: {} imgs, {} bboxes.'.format(len(ann_dict.keys()), cnt_objects))
        return data_infos

    def load_annotations(self, ann_file):
        """ Load annotation from list-style annotation file.
        list-style:
        # path img_height, img_width left top right bottom type
        vehicle_0000499.jpg 1080 1920 557 172 921 594 car
        ...

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.cat_ids = [i for i in range(len(self.CLASSES))]
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        data_infos = self._parse_meta(ann_file)
        self.img_ids = list(np.arange(len(data_infos)))
        return data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_info['ann']['bboxes'].shape[0] == 0:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def xyxy2xywh(self, bbox):
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

    def _proposal2list(self, results):
        """
        Convert detection results to hybrid-list style.

        # path left top right bottom category-id score
        """
        res_list = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            path = self.data_infos[img_id]['filename']
            bboxes = results[idx]
            label = 1
            for i in range(bboxes.shape[0]):
                left = float(bboxes[i][0])
                top = float(bboxes[i][1])
                right = float(bboxes[i][2])
                bottom = float(bboxes[i][3])
                score = float(bboxes[i][4])

                res_list.append([path, left, top, right, bottom, label, score])
        return res_list

    def _det2list(self, results):
        """
        Convert detection results to hybrid-list style.

        # path left top right bottom category-id score
        """
        res_list = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                path = self.data_infos[img_id]['filename']
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    left = float(bboxes[i][0])
                    top = float(bboxes[i][1])
                    right = float(bboxes[i][2])
                    bottom = float(bboxes[i][3])
                    score = float(bboxes[i][4])

                    res_list.append([path, left, top, right, bottom, label, score])
        return res_list

    def _dump_list(self, res_list, list_file):
        with open(list_file, 'w') as fout:
            fout.write('# path left top right bottom category-id score')
            for item in res_list:
                path, left, top, right, bottom, label, score = item
                fout.write('{} {} {} {} {} {} {}\n'.format(path, left, top, right, bottom, label, score))

    def results2file(self, results, outfile_prefix):
        """Dump the detection results to hybrid-list style.
        # path left top right bottom category-id score

        Args:
            results (list[list | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.txt",
                "somepath/xxx.proposal.txt".

        Returns:
            dict[str: str]: Possible keys are "bbox", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            res_list = self._det2list(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.txt'
            result_files['proposal'] = f'{outfile_prefix}.bbox.txt'
            self._dump_list(res_list, result_files['bbox'])
        elif isinstance(results[0], np.ndarray):
            res_list = self._proposal2list(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.txt'
            self._dump_list(res_list, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            img_id = self.img_ids[i]
            ann_info = self.data_info[img_id]['ann']
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                # if ann.get('ignore', False) or ann['iscrowd']:
                #     continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, save_dir=None, **kwargs):
        """Format the results to hybrid-list style

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            listfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the filepaths, tmp_dir is the temporal directory created \
                for saving json files when listfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if save_dir is None:
            tmp_dir = tempfile.TemporaryDirectory()
            listfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
            listfile_prefix = osp.join(save_dir, 'results')
        result_files = self.results2file(results, listfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
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
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
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
