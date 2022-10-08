# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.onnx.symbolic_helper import parse_args


class OpenVinoDetectionOutputOp(torch.autograd.Function):
    """Create DetectionOutput op.

    A dummy DetectionOutput operator for openvino end2end deployment.
    It will map to the DetectionOutput op of openvino. After converting
    to openvino, DetectionOutput op of openvino will get called
    automatically.
    
    openvino op `DetectionOutput`: https://docs.openvino.ai/latest/openvino_docs_ops_detection_DetectionOutput_1.html 
    > Inputs
        1 : 2D input tensor with box logits with shape [N, num_prior_boxes \* num_loc_classes \* 4] and type T. 
            num_loc_classes is equal to num_classes when share_location is 0 or it's equal to 1 otherwise. Required.

        2 : 2D input tensor with class predictions with shape [N, num_prior_boxes \* num_classes] and type T. Required.

        3 : 3D input tensor with proposals with shape [priors_batch_size, 1, num_prior_boxes \* prior_box_size] 
            or [priors_batch_size, 2, num_prior_boxes \* prior_box_size]. priors_batch_size is either 1 or N. 
            Size of the second dimension depends on variance_encoded_in_target. If variance_encoded_in_target is equal to 0, 
            the second dimension equals to 2 and variance values are provided for each boxes coordinates. 
            If variance_encoded_in_target is equal to 1, the second dimension equals to 1 and this tensor contains proposals 
            boxes only. prior_box_size is equal to 4 when normalized is set to 1 or it's equal to 5 otherwise. Required.

        4 : 2D input tensor with additional class predictions information described in the article. Its shape must be equal 
            to [N, num_prior_boxes \* 2]. Optional.

        5 : 2D input tensor with additional box predictions information described in the article. Its shape must be equal 
            to first input tensor shape. Optional.
        6:  attrs: The dictionary containing key, value pairs for attributes. Optional.
                    Available attributes are:
                    attrs = {
                        'num_classes': 1, 'keep_top_k': 200, 'top_k': 400,'nms_threshold': 0.45, 'normalized': True, 'clip_before_nms': True, 
                        'input_height': batch_height, 'input_width': batch_width, 'share_location': True, 'confidence_threshold': 0.009,
                        'clip_before_nms':False, 'clip_after_nms': False,
                    }
    > Outputs
        1 : 4D output tensor with type T. Its shape depends on keep_top_k or top_k being set. It keep_top_k[0] is 
            greater than zero, then the shape is [1, 1, N \* keep_top_k[0], 7]. If keep_top_k[0] is set to -1 and top_k is 
            greater than zero, then the shape is [1, 1, N \* top_k \* num_classes, 7]. Otherwise, the output shape is equal 
            to [1, 1, N \* num_classes \* num_prior_boxes, 7].

    > Types
        T : any supported floating-point type.

    C++ API:
    DetectionOutput(const Output<Node>& box_logits,
                    const Output<Node>& class_preds,
                    const Output<Node>& proposals,
                    const Output<Node>& aux_class_preds,
                    const Output<Node>& aux_box_preds,
                    const DetectionOutputAttrs& attrs);
       
    """

    @staticmethod
    def symbolic(g,
                 box_logits,
                 class_preds,
                 proposals,
                 aux_class_preds=None,
                 aux_box_preds=None,
                 num_classes_i=2, keep_top_k_i=200, top_k_i=400, background_label_id_i=0,
                 input_height_i=320, input_width_i=320,
                 nms_threshold_f=0.45, confidence_threshold_f=0.009,
                 normalized_i=False, share_location_i=True, clip_before_nms_i=False, clip_after_nms_i=False,
                 code_type_s="CENTER_SIZE", variance_encoded_in_target_i=1):
        """Symbolic function of dummy onnx DetectionOutput op for openvino."""
        return g.op(
            'custom_domain::DetectionOutput',
            box_logits,
            class_preds,
            proposals,
            aux_class_preds=aux_class_preds,
            aux_box_preds=aux_box_preds,
            num_classes_i=num_classes_i, keep_top_k_i=keep_top_k_i, top_k_i=top_k_i, background_label_id_i=background_label_id_i,
            input_height_i=input_height_i, input_width_i=input_width_i,
            nms_threshold_f=nms_threshold_f, confidence_threshold_f=confidence_threshold_f,
            normalized_i=normalized_i, share_location_i=share_location_i, clip_before_nms_i=clip_before_nms_i, clip_after_nms_i=clip_after_nms_i,
            code_type_s=code_type_s, variance_encoded_in_target_i=variance_encoded_in_target_i)

    @staticmethod
    def forward(ctx,
                box_logits,
                class_preds,
                proposals,
                aux_class_preds=None,
                aux_box_preds=None,
                num_classes_i=2, keep_top_k_i=200, top_k_i=400, background_label_id_i=0,
                input_height_i=320, input_width_i=320,
                nms_threshold_f=0.45, confidence_threshold_f=0.009,
                normalized_i=False, share_location_i=True, clip_before_nms_i=False, clip_after_nms_i=False,
                code_type_s="CENTER_SIZE", variance_encoded_in_target_i=1):
        """Forward function of dummy onnx DetectionOutput op for openvino."""
        return torch.rand(1, keep_top_k_i, 7)


openvino_detection_output_forward = OpenVinoDetectionOutputOp.apply
