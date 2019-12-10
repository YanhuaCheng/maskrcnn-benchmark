"""
This file contains specific functions for computing losses on the RetinaNet
file
"""
import logging

import torch
from maskrcnn_benchmark.layers import SigmoidFocalLoss, smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.rpn.loss import RPNLossComputation
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, cat_boxlist
from torch.nn import functional as F

from ..utils import concat_box_prediction_layers


class RetinaNetLossComputation(object):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, proposal_matcher, box_coder,
                 generate_labels_func,
                 sigmoid_focal_loss,
                 bbox_reg_beta=0.11,
                 regress_norm=1.0,
                 use_ignored_bbox = False,
                 classify_holistic_image=False,
                 num_classes=None,
                 num_holistic_classes=None,
                 classify_objectness_image=False,
                 objectness_norm=1.0,
                 fg_bg_sampler=None):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.box_cls_loss_func = sigmoid_focal_loss
        self.bbox_reg_beta = bbox_reg_beta
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        self.regress_norm = regress_norm
        self.use_ignored_bbox = use_ignored_bbox
        self.classify_holistic_image = classify_holistic_image
        self.num_classes = num_classes
        self.num_holistic_classes = num_holistic_classes
        self.classify_objectness_image = classify_objectness_image
        self.objectness_norm = objectness_norm
        self.fg_bg_sampler = fg_bg_sampler
        self.logger = logging.getLogger(__name__)

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # discard all bboxes for noisy images, which are only used for holistic image classification
            if self.classify_holistic_image:
                bbox_labels = targets_per_image.get_field('labels')
                if bbox_labels.max() >= self.num_classes:
                    assert(bbox_labels.min() >= self.num_classes), "nosiy images should be labeled from {} to {}".format(self.num_classes, self.num_holistic_classes)
                    labels_per_image.fill_(-2)
                else:
                    assert(bbox_labels.min() > 0), "positive images should be labeled from {} to {}".format(1, self.num_classes-1)

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, anchors, box_cls, box_regression, objectness_cls, targets):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            retinanet_cls_loss (Tensor)
            retinanet_regression_loss (Tensor)
        """
        device = box_cls[0].device
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        if self.classify_objectness_image:
            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
            sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
            sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        N = len(labels)
        box_cls, box_regression, objectness_cls = \
                concat_box_prediction_layers(box_cls, box_regression, objectness_cls)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)

        if pos_inds.numel() > 0:
            retinanet_regression_loss = smooth_l1_loss(
                box_regression[pos_inds],
                regression_targets[pos_inds],
                beta=self.bbox_reg_beta,
                size_average=False,
            ) / (max(1, pos_inds.numel() * self.regress_norm))
        else:
            retinanet_regression_loss = torch.tensor(0.0, device=device)
            self.logger.info("This batch has none positive anchors for bbox regression")

        if self.use_ignored_bbox:
            labels = labels.int()
            retinanet_cls_loss = self.box_cls_loss_func(
                box_cls,
                labels
            ) / (pos_inds.numel() + N)
        else:
            valid_inds1 = torch.nonzero(labels >= 0).squeeze(1)
            valid_inds2 = torch.nonzero(labels < -1).squeeze(1)
            valid_inds = torch.cat([valid_inds1, valid_inds2], dim=0)
            labels = labels.int()
            if valid_inds.numel() > 0:
                retinanet_cls_loss = self.box_cls_loss_func(
                    box_cls[valid_inds],
                    labels[valid_inds]
                ) * 1000 / (max(1, valid_inds.numel()))
            else:
                retinanet_cls_loss = torch.tensor(0.0, device=device)
                self.logger.info("This batch has none valid anchors for bbox classification")

        if self.classify_objectness_image:
            objectness_labels = labels >= 1
            objectness_labels = objectness_labels.view(-1, 1)
            objectness_labels = objectness_labels.float()
            retinanet_objectness_loss = F.binary_cross_entropy_with_logits(
                objectness_cls[sampled_inds],
                objectness_labels[sampled_inds],
                reduction='sum'
            ) / (sampled_inds.numel() * self.objectness_norm)
        else:
            retinanet_objectness_loss = torch.tensor(0.0, device=device)

        return retinanet_cls_loss, retinanet_regression_loss, retinanet_objectness_loss

class RetinaNetHolisticLossComputation(object):
    """
    This class computes the RetinaNet loss for holistic image classification.
    """

    def __init__(self, class_weights,
                 holistic_classify_norm=1.0,
                 use_focal_loss=False,
                 alpha=0.25, gamma=2.0):
        """
        Arguments:
            class_weights
            holistic_classify_norm
        """
        self.class_weights = class_weights
        self.holistic_classify_norm = holistic_classify_norm
        self.holistic_cls_loss_func = F.binary_cross_entropy_with_logits
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, holistic_cls, targets):
        """
        Arguments:
            holistic_cls (Tensor)
            targets (list[BoxList])

        Returns:
            retinanet_holistic_cls_loss (Tensor)
        """
        device = holistic_cls.device
        holitic_labels = []
        for targets_per_image in targets:
            onehot_per_image = torch.zeros((1, len(self.class_weights)))
            labels_per_image = targets_per_image.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64).to('cpu') - 1
            onehot_per_image.scatter_(1, labels_per_image.view(1, -1), 1.0)
            holitic_labels.append(onehot_per_image.to(device))

        holitic_labels = torch.cat(holitic_labels, dim=0)
        N = holitic_labels.shape[0]

        if not self.use_focal_loss:
            retinanet_holistic_cls_loss = self.holistic_cls_loss_func(
                holistic_cls,
                holitic_labels,
                reduction='sum',
                pos_weight=torch.tensor(self.class_weights, device=device)
            ) / (N * self.holistic_classify_norm)
        else:
            bce_loss = self.holistic_cls_loss_func(
                holistic_cls,
                holitic_labels,
                reduction='none',
                pos_weight=torch.tensor(self.class_weights, device=device)
            )
            pt = torch.exp(-bce_loss)
            retinanet_holistic_cls_loss = self.alpha * (1-pt)**self.gamma * bce_loss
            retinanet_holistic_cls_loss = torch.sum(retinanet_holistic_cls_loss) / (N * self.holistic_classify_norm)

        return retinanet_holistic_cls_loss

def generate_retinanet_labels(matched_targets):
    labels_per_image = matched_targets.get_field("labels")
    return labels_per_image


def make_retinanet_loss_evaluator(cfg, box_coder):
    classify_holistic_image = cfg.MODEL.CLASSIFY_HOLISTIC_IMAGE
    classify_objectness_image = cfg.MODEL.CLASSIFY_OBJECTNESS_IMAGE
    num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
    num_holistic_classes = cfg.MODEL.RETINANET.NUM_HOLISTIC_CLASSES

    matcher = Matcher(
        cfg.MODEL.RETINANET.FG_IOU_THRESHOLD,
        cfg.MODEL.RETINANET.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )
    sigmoid_focal_loss = SigmoidFocalLoss(
        cfg.MODEL.RETINANET.LOSS_GAMMA,
        cfg.MODEL.RETINANET.LOSS_ALPHA
    )
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RetinaNetLossComputation(
        matcher,
        box_coder,
        generate_retinanet_labels,
        sigmoid_focal_loss,
        bbox_reg_beta = cfg.MODEL.RETINANET.BBOX_REG_BETA,
        regress_norm = cfg.MODEL.RETINANET.BBOX_REG_WEIGHT,
        use_ignored_bbox = cfg.MODEL.RETINANET.LOSS_USE_IGNORE,
        classify_holistic_image = classify_holistic_image,
        num_classes = num_classes,
        num_holistic_classes = num_holistic_classes,
        classify_objectness_image = classify_objectness_image,
        objectness_norm = cfg.MODEL.RETINANET.OBJECTNESS_LOSS_WEIGHT,
        fg_bg_sampler = fg_bg_sampler,
    )

    if classify_holistic_image:
        class_weights = cfg.MODEL.RETINANET.HOLISTIC_CLASS_WEIGHT
        holistic_classify_norm = cfg.MODEL.RETINANET.HOLISTIC_LOSS_WEIGHT
        use_focal_loss = cfg.MODEL.RETINANET.HOLISTIC_USE_FOCAL_LOSS,
        alpha = cfg.MODEL.RETINANET.HOLISTIC_ALPHA
        gamma = cfg.MODEL.RETINANET.HOLISTIC_GAMMA
        if len(class_weights) == 1:
            class_weights = class_weights * (cfg.MODEL.RETINANET.NUM_HOLISTIC_CLASSES - 1)
        loss_holistic_evaluator = RetinaNetHolisticLossComputation(class_weights, holistic_classify_norm, use_focal_loss, alpha, gamma)
    else:
        loss_holistic_evaluator = None

    return loss_evaluator, loss_holistic_evaluator
