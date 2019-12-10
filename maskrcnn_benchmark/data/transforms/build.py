# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        keep_aspect_ratio = cfg.INPUT.KEEP_ASPECT_RATIO
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
        keep_aspect_ratio = cfg.INPUT.KEEP_ASPECT_RATIO
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
            T.RandomValidAreaCrop(crop_prob),
            T.Resize(min_size, max_size, keep_aspect_ratio),
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

def build_transforms_batch(cfg, is_train=True):
    if is_train:
        keep_aspect_ratio = cfg.INPUT.KEEP_ASPECT_RATIO
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = cfg.INPUT.FLIP_PROB_TRAIN
        vertical_flip_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        flip_90_prob = cfg.INPUT.FLIP_90_PROB_TRAIN
        crop_prob = cfg.INPUT.CROP_PROB_TRAIN
        blur_prob = cfg.INPUT.BLUR_PROB_TRAIN
        contrast_prob = cfg.INPUT.CONTRAST_PROB_TRAIN
        hsv_prob = cfg.INPUT.HSV_PROB_TRAIN
        mixup_prob = cfg.INPUT.MIXUP_PROB_TRAIN
        mixup_alpha = cfg.INPUT.MIXUP_ALPHA
        aug_bndobj_prob = cfg.INPUT.AUG_BNDOBJ_PROB_TRAIN
        aug_bndobj_ratio = cfg.INPUT.AUG_BNDOBJ_RATIO
    else:
        keep_aspect_ratio = cfg.INPUT.KEEP_ASPECT_RATIO
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0
        vertical_flip_prob = 0
        flip_90_prob = 0
        crop_prob = 0
        blur_prob = 0
        contrast_prob = 0
        hsv_prob = 0
        mixup_prob = 0
        mixup_alpha = cfg.INPUT.MIXUP_ALPHA
        aug_bndobj_prob = 0
        aug_bndobj_ratio = cfg.INPUT.AUG_BNDOBJ_RATIO

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform_batch = T.NormalizeBatch(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )


    transform_img = T.Compose(
        [
            T.RandomValidAreaCrop(crop_prob),
            T.Blur(blur_prob),
            T.ContrastNormalization(contrast_prob),
            T.ChangeHSV(hsv_prob),
            T.RandomHorizontalFlip(flip_prob),
            T.RandomVerticalFlip(vertical_flip_prob),
            T.RandomRotate90(flip_90_prob),
        ]
    )
    transform_batch = T.Compose(
        [
            T.ResizeBatch(min_size, max_size, keep_aspect_ratio),
            T.MixUpBatch(mixup_prob, mixup_alpha),
            T.AugBndObjBatch(aug_bndobj_prob, aug_bndobj_ratio),
            T.ToTensorBatch(),
            normalize_transform_batch,
        ]
    )
    return transform_img, transform_batch
