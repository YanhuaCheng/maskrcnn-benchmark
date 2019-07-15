# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import pdb

import numpy as np

import cv2
import torch
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from torchvision import transforms as T


class COCODemo(object):
    # COCO categories for pretty print
    CATEGORIES = ["__background", 'shoes','bag','make-ups','clothes','HEA','toy_music','book','food','ACC','furniture','others']

    def __init__(
        self,
        cfg,
        data_source = 0,
        min_image_size=512,
        score_weight=0.6,
        pos_weight=0.3,
        area_weight=0.1,
        min_bbox_w=0.0,
        min_bbox_h=0.0,
        min_bbox_area=0.0,
        topn=1,
        visualize_flag = True
    ):
        self.cfg = cfg.clone()
        self.data_source = data_source
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size
        self.score_weight = score_weight
        self.pos_weight = pos_weight
        self.area_weight = area_weight
        self.min_bbox_w = min_bbox_w
        self.min_bbox_h = min_bbox_h
        self.min_bbox_area = min_bbox_area
        self.topn = topn
        self.visualize_flag = visualize_flag

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        self.cpu_device = torch.device("cpu")

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image, topn=None):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        if topn is not None:
           self.topn = topn
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions, image)

        if self.visualize_flag:
           result = image.copy()
           result = self.overlay_boxes(result, top_predictions)
           return result
        else:
           return top_predictions

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        return prediction

    def select_top_predictions(self, predictions, image):
        """
        Select only predictions which have the highest lambda_score*score + lambda_position*position

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        height, width = image.shape[:2]
        scores = predictions.get_field("scores")
        boxes = predictions.bbox
        weighted_scores = []
        for idx, (box, score) in enumerate(zip(boxes, scores)):
            box = box.to(torch.int64)
            x1, y1, x3, y3 = box[:4].tolist()
            if (y3 - y1 > self.min_bbox_h * height) and (x3 - x1 > self.min_bbox_w * width) and ((x3 - x1) * (y3 - y1) > self.min_bbox_area * height * width):
               weighted_score = self.score_weight * score.item() + self.pos_weight * math.exp(-5*(math.pow(0.5*(x1+x3)/width-0.5, 2)+math.pow(0.5*(y1+y3)/height-0.5, 2))) + self.area_weight * (x3 - x1) * (y3 - y1) / (height * width)
               #print('weihted_score={}={}*{}(score)+{}*{}(position)+{}*{}(area)'.format(weighted_score, self.score_weight, score.item(), self.pos_weight, math.exp(-5*(math.pow(0.5*(x1+x3)/width-0.5, 2)+math.pow(0.5*(y1+y3)/height-0.5, 2))), self.area_weight, (x3 - x1) * (y3 - y1) / (height * width)))
            else:
               weighted_score = 0.0
            weighted_scores.append(weighted_score)
        rank_idxs = np.argsort(-1*np.array(weighted_scores))
        top_predictions = []
        for rank_idx in rank_idxs[:self.topn]:
            if weighted_scores[rank_idx] <= 0:
                continue
            top_prediction = dict()
            scores_all = predictions[[rank_idx]].get_field("scores_all")
            scores_all[:, -1] = max(scores_all[:, -1], self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_HIGH) #other category
            boxes = predictions[[rank_idx]].bbox
            box = boxes[0].to(torch.int64)
            top_prediction['bbox'] = box[:4].tolist()
            if scores_all.shape[1] < len(self.CATEGORIES):
                scores, labels = scores_all[0][0:].sort(0, descending=True)
            else:
                scores, labels = scores_all[0][1:].sort(0, descending=True)
            top_prediction['category'] = [self.CATEGORIES[label+1] for label in labels]
            top_prediction['score'] = [score.item() for score in scores]
            top_predictions.append(top_prediction)
        if len(top_predictions) == 0 and self.data_source == 0: #use whole image if no bbox
            top_prediction = dict()
            top_prediction['bbox'] = [0, 0, width-1, height-1]
            top_prediction['category'] = [self.CATEGORIES[-1]]
            top_prediction['score'] = [0.0]
            top_predictions.append(top_prediction)
        return top_predictions

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        for prediction in predictions:
            template = ''
            for idx in range(min(3, len(prediction['score']))):
               template += "{}:{:.2f},".format(prediction['category'][idx], prediction['score'][idx])
            color = [0, 0, 255]
            top_left, bottom_right = prediction['bbox'][:2], prediction['bbox'][2:]
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )
            cv2.putText(
                image, template, (top_left[0], top_left[1]+15), cv2.FONT_HERSHEY_SIMPLEX, .6, tuple(color), 1
            )

        return image
