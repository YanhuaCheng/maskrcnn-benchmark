# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import pdb
import numpy as np
import cv2
import torch
import transforms as T
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from torchvision import transforms as T


class COCODemo(object):
    # COCO categories for pretty print
    CATEGORIES = ["__background", 'shoes','bag','make-ups','clothes','HEA','toy_music','book','food','ACC','furniture','others']

    def __init__(
        self,
        cfg
    ):
        self.cfg = cfg.clone()
        self.score_weight = cfg.DEPLOY.SCORE_WEIGHT
        self.pos_weight = cfg.DEPLOY.POS_WEIGHT
        self.area_weight = cfg.DEPLOY.AREA_WEIGHT
        self.min_bbox_w = cfg.DEPLOY.MIN_BBOX_W
        self.min_bbox_h = cfg.DEPLOY.MIN_BBOX_H
        self.min_bbox_area = cfg.DEPLOY.MIN_BBOX_AREA
        self.topn = cfg.DEPLOY.TOPN
        self.visualize_flag = cfg.DEPLOY.VISUALIZE
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cpu_device = torch.device("cpu")
        self.model.to(self.device)
        self.unnormalize = T.UnNormalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=cfg.OUTPUT_DIR)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)    

    def get_top_detections(self, imgs, imgs_h, imgs_w, imgs_name, topn=None):
        """
        Arguments:
            imgs, imgs_idx: images and idxs as returned by data_loader
            topn: max number bboxs per image
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        if topn is not None:
           self.topn = topn

        top_predictions = self.compute_prediction(imgs, imgs_h, imgs_w)       

        if self.visualize_flag:
           save_dir = os.path.join(self.cfg.OUTPUT_DIR, os.path.split(cfg.MODEL.WEIGHT)[1])
           if not os.path.isdir(save_dir):
              os.makedirs(save_dir)
           for img, img_name, img_h, img_w, top_prediction in zip(imgs, imgs_name, imgs_h, imgs_w, top_predictions):
               img_result = self.unnormalize(img)
               img_result = cv2.resize(img_result.item(), (img_w, img_h))
               img_result = self.overlay_boxes(np.asarray(img_result, np.uint8), top_prediction)
               cv2.imwrite("{}/{}.jpg".format(save_dir, img_name.replace('/', '_')), img_result)
        return top_predictions

    def compute_prediction(self, imgs, imgs_h, imgs_w):
        """
        Arguments:
            imgs, imgs_h, imgs_w (np.ndarray): imgs as returned by data_loader

        Returns:
            top_predictions ([BoxList]): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        top_predictions = []
        imgs_list = to_image_list(imgs, cfg.DATALOADER.SIZE_DIVISIBILITY)
        imgs_list = imgs_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(imgs_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        for prediction, img, img_h, img_w in zip(predictions, imgs, imgs_h, imgs_w):
           # reshape prediction (a BoxList) into the original image size           
           prediction = prediction.resize((img_w, img_h))
           top_prediction = select_top_predictions(self, predictions, img_w, img_h):
           top_predictions.append(top_prediction)
        return top_predictions

    def select_top_predictions(self, prediction, img_w, img_h):
        """
        Select only predictions which have the highest lambda_score*score + lambda_position*position

        Arguments:
            prediction (BoxList): the result of the computation by the model.
                It should contain the field `scores`.
            img_w, img_h: img shape
        Returns:
            top_predictions ([dict]): the detected objects with bbox position
        """
        scores = prediction.get_field("scores")
        boxes = prediction.bbox
        weighted_scores = []
        for idx, (box, score) in enumerate(zip(boxes, scores)):
            box = box.to(torch.int64)
            x1, y1, x3, y3 = box[:4].tolist()
            if (y3 - y1 > self.min_bbox_h * img_h) and (x3 - x1 > self.min_bbox_w * img_w) and ((x3 - x1) * (y3 - y1) > self.min_bbox_area * img_h * img_w):
               weighted_score = self.score_weight * score.item() + self.pos_weight * math.exp(-5*(math.pow(0.5*(x1+x3)/img_w-0.5, 2)+math.pow(0.5*(y1+y3)/img_h-0.5, 2))) + self.area_weight * (x3 - x1) * (y3 - y1) / (img_h * img_w)
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
            scores_all = prediction[[rank_idx]].get_field("scores_all")
            scores_all[:, -1] = max(scores_all[:, -1], self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_HIGH) #other category
            boxes = prediction[[rank_idx]].bbox
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