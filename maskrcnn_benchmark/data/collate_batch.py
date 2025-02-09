# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.visualize import visual_transforms


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0, transforms_batch=None):
        self.size_divisible = size_divisible
        self.transforms_batch = transforms_batch

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = transposed_batch[0]
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        if self.transforms_batch is not None:
           images, targets = self.transforms_batch(images, targets)
        # visual_transforms(images, targets)
        # exit(0)
        images = to_image_list(images, self.size_divisible)
        return images, targets, img_ids


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))
