# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def forward(self, img, img_metas, return_loss=True, single_output=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            if single_output:
                return self.onnx_export_single(img[0], img_metas[0])
            else:
                return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.backbone(img)
        return x
        # x = self.extract_feat(img)
        # outs = self.bbox_head(x)
        # # get origin input shape to support onnx dynamic shape

        # # get shape as tensor
        # img_shape = torch._shape_as_tensor(img)[2:]
        # img_metas[0]['img_shape_for_onnx'] = img_shape
        # # get pad input shape to support onnx dynamic shape for exporting
        # # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # # for inference
        # img_metas[0]['pad_shape_for_onnx'] = img_shape

        # if len(outs) == 2:
        #     # add dummy score_factor
        #     outs = (*outs, None)
        # # TODO Can we change to `get_bboxes` when `onnx_export` fail
        # det_bboxes, det_labels = self.bbox_head.onnx_export(
        #     *outs, img_metas, with_nms=with_nms)

        # return det_bboxes, det_labels

    def onnx_export_single(self, img, img_metas, with_nms=True):
        det_bboxes, det_labels = self.onnx_export(img, img_metas, with_nms)
        '''
        image_ids.shape
            torch.Size([1, 200, 1])
        det_labels.shape
            torch.Size([1, 200, 1])
        det_bboxes.shape
            torch.Size([1, 200, 5])
        '''
        batch, _, input_h, input_w = img.shape
        det_labels = det_labels.type_as(det_bboxes)

        '''
        positive category id should start from 1.
        '''
        labels = det_labels.reshape(1, det_bboxes.shape[0], det_bboxes.shape[1], 1)
        labels[..., 0] = labels[..., 0] + 1

        assert batch == 1, 'unsupported batchsize: {}'.format(batch)
        image_ids = torch.full_like(labels, 0)

        scores = det_bboxes[:, :, -1].reshape(1, det_bboxes.shape[0], det_bboxes.shape[1], 1)

        '''
        bounding boxes range to [0, 1]
        '''
        bboxes = det_bboxes[:, :, :4].reshape(1, det_bboxes.shape[0], det_bboxes.shape[1], 4)
        bboxes[..., 0] = bboxes[..., 0] / input_w
        bboxes[..., 1] = bboxes[..., 1] / input_h
        bboxes[..., 2] = bboxes[..., 2] / input_w
        bboxes[..., 3] = bboxes[..., 3] / input_h

        '''
        [image_id, label, conf, x_min, y_min, x_max, y_max]
        '''
        output = torch.cat([image_ids, labels, scores, bboxes], -1)
        return output
