import math

import torch
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from torch import nn

from ..anchor_generator import make_anchor_generator_retinanet
from .inference import make_retinanet_postprocessor
from .loss import make_retinanet_loss_evaluator


class RetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
                        * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels,  num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )

        # Initialization
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits,
                  self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)


        # retinanet_bias_init
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        # holistic image classification
        self.classify_holistic_image = cfg.MODEL.CLASSIFY_HOLISTIC_IMAGE
        if self.classify_holistic_image:
            num_holistic_classes = cfg.MODEL.RETINANET.NUM_HOLISTIC_CLASSES - 1
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            holistic_cls_logits = []
            holistic_cls_logits.append(nn.Linear(in_channels, in_channels))
            holistic_cls_logits.append(nn.ReLU())
            holistic_cls_logits.append(nn.Linear(in_channels, num_holistic_classes))
            self.add_module('holistic_cls_logits', nn.Sequential(*holistic_cls_logits))
            for l in self.holistic_cls_logits.modules():
                if isinstance(l, nn.Linear):
                    nn.init.normal_(l.weight, std=0.01)
                    nn.init.constant_(l.bias, 0)

        # objectness classification
        self.classify_objectness_image = cfg.MODEL.CLASSIFY_OBJECTNESS_IMAGE
        if self.classify_objectness_image:
            self.objectness_logits = nn.Conv2d(
                in_channels, num_anchors, kernel_size=3, stride=1,
                padding=1
            )
            for l in self.objectness_logits.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
            torch.nn.init.constant_(self.objectness_logits.bias, bias_value)


    def forward(self, x):
        logits = []
        bbox_reg = []
        holistic_logits = []
        objectness_logits = []
        for feature in x:
            cls_feature = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_feature))
            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
            if self.classify_objectness_image:
                objectness_logits.append(self.objectness_logits(cls_feature))
            if self.classify_holistic_image:
                feature = self.avgpool(feature)
                feature = feature.view(feature.size(0), -1)
                holistic_logits.append(self.holistic_cls_logits(feature))
        if self.classify_holistic_image:
            holistic_logits = sum(holistic_logits)
        return logits, bbox_reg, holistic_logits, objectness_logits


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(RetinaNetModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator_retinanet(cfg)
        head = RetinaNetHead(cfg, in_channels)
        box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        box_selector_test = make_retinanet_postprocessor(cfg, box_coder, is_train=False)

        loss_evaluator, loss_holistic_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.loss_holistic_evaluator = loss_holistic_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, holistic_cls, objectness_cls = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, box_cls, box_regression, holistic_cls, objectness_cls, targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression, holistic_cls, objectness_cls)

    def _forward_train(self, anchors, box_cls, box_regression, holistic_cls, objectness_cls, targets):

        loss_box_cls, loss_box_reg, loss_objectness_cls = self.loss_evaluator(
            anchors, box_cls, box_regression, objectness_cls, targets
        )
        if self.loss_holistic_evaluator is None:
            loss_holistic_cls = torch.tensor(0.0, device=loss_box_cls.device)
        else:
            loss_holistic_cls = self.loss_holistic_evaluator(
                holistic_cls, targets
            )
        losses = {
            "loss_retina_cls": loss_box_cls,
            "loss_retina_reg": loss_box_reg,
            "loss_retina_holistic_cls": loss_holistic_cls,
            "loss_retina_objectness_cls": loss_objectness_cls,
        }
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression, holistic_cls, objectness_cls):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        results = {'boxes': boxes, 'holistic_cls': holistic_cls}
        return results, {}


def build_retinanet(cfg, in_channels):
    return RetinaNetModule(cfg, in_channels)
