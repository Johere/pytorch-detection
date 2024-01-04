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
class CustomListDataset(CustomDataset):
    """
    list_file for each line: only contains image-path
    Annotations will be dummy data

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

    def _parse_meta(self, list_file):
        """
        # path
        MVI_40855/img00005.jpg
        """
        print('------------------------------------------------------------')
        print('start parsing meta files: {}'.format(list_file))
        data_infos = []
        with open(list_file, 'r') as f:
            lines = f.readlines()

        img_list = []
        for ln in lines[1:]:
            if ln.startswith("#"):
                continue
            ln = ln.strip()
            filename = ln.split()[0]
            img_list.append(filename)

        for filename in img_list:

            cur_data_info = {
                "filename": filename,
                "width": 0,
                "height": 0
            }

            ann = dict(
                bboxes=np.zeros((0, 4), dtype=np.float32),
                labels=np.array([], dtype=np.int64),
                bboxes_ignore=np.zeros((0, 4), dtype=np.float32)
            )
            cur_data_info.update({'ann': ann})

            data_infos.append(cur_data_info)
        print('------------------------------------------------------------')
        print('parsing custom list files done. Got: {} imgs.'.format(len(data_infos)))
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
            fout.write('# path left top right bottom category-id score\n')
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
                 **kwargs):
        raise NotImplementedError("Custom list dataset do not support for evaluation!")
