# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import numpy as np
from PIL import Image

import imgaug as ia
import torch
import torchvision
from imgaug import augmenters as iaa
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target

class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target

class Blur(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
           aug_seq = iaa.OneOf([
                 iaa.GaussianBlur(sigma=(0.0, 1.0)),
                 iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                 iaa.Add((-40, 40), per_channel=0.5),
                 iaa.JpegCompression(compression=(5, 25))
           ])
           image = aug_seq.augment_image(np.array(image))
           image = Image.fromarray(np.uint8(image))
        return image, target

class ContrastNormalization(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
           aug_seq = iaa.ContrastNormalization((0.5, 1.5))
           image = aug_seq.augment_image(np.array(image))
           image = Image.fromarray(np.uint8(image))
        return image, target

class RandomCrop(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
           w, h = image.size
           bbox = [random.randint(0, int(w*0.1)), random.randint(0, int(h*0.1)), random.randint(int(w*0.9), w), random.randint(int(h*0.9), h)]
           image = image.crop(bbox)
           target = target.crop(bbox)
        return image, target

class ChangeHSV(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
           aug_seq = iaa.OneOf([
                     iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB", children=iaa.WithChannels(0, iaa.Add((10, 50)))),
                     iaa.Grayscale(alpha=(0.0, 1.0))
           ])
           image = aug_seq.augment_image(np.array(image))
           image = Image.fromarray(np.uint8(image))
        return image, target

class RandomRotate90(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.rotate(image, -90, expand=True)
            target = target.transpose(2)
        return image, target
