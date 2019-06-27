# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        
		
        flip_prob = cfg.INPUT.FLIP_PROB_TRAIN
		vertical_flip_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        flip_90_prob = cfg.INPUT.FLIP_90_PROB_TRAIN
        crop_prob = cfg.INPUT.CROP_PROB_TRAIN
        blur_prob = cfg.INPUT.BLUR_PROB_TRAIN
        contrast_prob = cfg.INPUT.CONTRAST_PROB_TRAIN
        hsv_prob = cfg.INPUT.HSV_PROB_TRAIN

    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0
        vertical_flip_prob = 0
        flip_90_prob = 0
        crop_prob = 0
        blur_prob = 0
        contrast_prob = 0
        hsv_prob = 0        

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )


    transform = T.Compose(
        [
            T.RandomCrop(crop_prob),
            T.Resize(min_size, max_size),
            T.Blur(blur_prob),
            T.ContrastNormalization(contrast_prob),
            T.ChangeHSV(hsv_prob),
            T.RandomHorizontalFlip(flip_prob),
            T.RandomVerticalFlip(vertical_flip_prob),
            T.RandomRotate90(flip_90_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
