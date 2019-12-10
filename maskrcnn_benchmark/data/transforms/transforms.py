# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import numpy as np
from PIL import Image

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
    def __init__(self, min_size, max_size, keep_aspect_ratio):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.keep_aspect_ratio = keep_aspect_ratio

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        if self.keep_aspect_ratio is False:
            size = random.randint(self.min_size[0], self.min_size[-1])
            return (size, size)
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

class ResizeBatch(object):
    def __init__(self, min_size, max_size, keep_aspect_ratio):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.keep_aspect_ratio = keep_aspect_ratio
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size, size):
        w, h = image_size
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

    def __call__(self, images, targets):
        assert(len(images) == len(targets))
        images = list(images)
        targets = list(targets)
        if self.keep_aspect_ratio is False:
            min_size = random.randint(self.min_size[0], self.min_size[-1])
        else:
            min_size = random.choice(self.min_size)
        for idx in range(len(images)):
            if self.keep_aspect_ratio is False:
                size = (min_size, min_size)
            else:
                size = self.get_size(images[idx].size, min_size)
            images[idx] = F.resize(images[idx], size)
            targets[idx] = targets[idx].resize(images[idx].size)
        return tuple(images), tuple(targets)

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

class ToTensorBatch(object):
    def __call__(self, images, targets):
        assert(len(images) == len(targets))
        images = list(images)
        targets = list(targets)
        for idx in range(len(images)):
            images[idx] = F.to_tensor(images[idx])
        return tuple(images), tuple(targets)

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

class NormalizeBatch(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, images, targets):
        assert(len(images) == len(targets))
        images = list(images)
        targets = list(targets)
        for idx in range(len(images)):
            if self.to_bgr255:
                images[idx] = images[idx][[2, 1, 0]] * 255
            images[idx] = F.normalize(images[idx], mean=self.mean, std=self.std)
        return tuple(images), tuple(targets)

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
           box = [random.randint(0, int(w*0.1)), random.randint(0, int(h*0.1)), random.randint(int(w*0.9), w), random.randint(int(h*0.9), h)]
           image = image.crop(box)
           target = target.crop(box)
        return image, target

class RandomValidAreaCrop(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
           target, box = target.valid_area_crop()
           image = image.crop(box)
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

class RandomRotate90Batch(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, images, targets):
        if random.random() < self.prob:
            assert(len(images) == len(targets))
            images = list(images)
            targets = list(targets)
            for idx in range(len(images)):
                images[idx] = F.rotate(images[idx], -90, expand=True)
                targets[idx] = targets[idx].transpose(2)
        return tuple(images), tuple(targets)

class MixUpBatch(object):
    def __init__(self, prob=0.5, alpha=[0.0, 1.0]):
        self.prob = prob
        self.alpha = alpha

    def __call__(self, images, targets):
        assert(len(images) == len(targets))
        images = list(images)
        images_blend = []
        for idx in range(len(images)):
            if random.random() < self.prob:
                alpha = random.uniform(self.alpha[0], self.alpha[-1])
                img1 = images[idx].convert("RGBA")
                img2 = images[(idx + random.randint(1, len(images)-1)) % len(images)].convert("RGBA")
                img3 = Image.blend(img1, img2, alpha)
                images_blend.append(img3.convert("RGB"))
            else:
                images_blend.append(images[idx])
        return tuple(images_blend), targets

class AugBndObjBatch(object):
    def __init__(self, prob=0.5, ratio=[0.0, 0.5]):
        self.prob = prob
        self.ratio = ratio

    def __call__(self, images, targets):
        assert(len(images) == len(targets))
        images = list(images)
        targets = list(targets)
        images_blend = []
        for idx in range(len(images)):
            if random.random() < self.prob:
                target1_bbox, img1_size = targets[idx].valid_area(union_flag=True)
                img1 = images[idx].copy()
                img2_idx = (idx + random.randint(1, len(images)-1)) % len(images)
                target2_bbox, img2_size = targets[img2_idx].valid_area(union_flag=False)
                target2_bbox_w = target2_bbox[2] - target2_bbox[0]
                target2_bbox_h = target2_bbox[3] - target2_bbox[1]
                img2 = images[img2_idx]
                pos = random.randint(0, 3)
                if pos == 0:  # left
                    crop_w = int(min(target2_bbox_w * random.uniform(self.ratio[0], self.ratio[-1]), target1_bbox[0]))
                    crop_h = int(min(target2_bbox_h, img1_size[1]))
                    if crop_w > 0 and crop_h > 0:
                        img2_crop = img2.crop((target2_bbox[2]-crop_w, target2_bbox[3]-crop_h, target2_bbox[2], target2_bbox[3]))
                        paste_x = 0
                        paste_y = random.randint(0, img1_size[1]-crop_h)
                        img1.paste(img2_crop, (paste_x, paste_y, paste_x+crop_w, paste_y+crop_h))
                elif pos == 1:  # top
                    crop_w = int(min(target2_bbox_w, img1_size[0]))
                    crop_h = int(min(target2_bbox_h * random.uniform(self.ratio[0], self.ratio[-1]), target1_bbox[1]))
                    if crop_w > 0 and crop_h > 0:
                        img2_crop = img2.crop((target2_bbox[2]-crop_w, target2_bbox[3]-crop_h, target2_bbox[2], target2_bbox[3]))
                        paste_x = random.randint(0, img1_size[0]-crop_w)
                        paste_y = 0
                        img1.paste(img2_crop, (paste_x, paste_y, paste_x+crop_w, paste_y+crop_h))
                elif pos == 2:  # right
                    crop_w = int(min(target2_bbox_w * random.uniform(self.ratio[0], self.ratio[-1]), img1_size[0]-target1_bbox[2]))
                    crop_h = int(min(target2_bbox_h, img1_size[1]))
                    if crop_w > 0 and crop_h > 0:
                        img2_crop = img2.crop((target2_bbox[0], target2_bbox[1], target2_bbox[0]+crop_w, target2_bbox[1]+crop_h))
                        paste_x = img1_size[0]
                        paste_y = random.randint(0, img1_size[1]-crop_h)
                        img1.paste(img2_crop, (paste_x-crop_w, paste_y, paste_x, paste_y+crop_h))
                else:  # bottom
                    crop_w = int(min(target2_bbox_w, img1_size[0]))
                    crop_h = int(min(target2_bbox_h * random.uniform(self.ratio[0], self.ratio[-1]), img1_size[1]-target1_bbox[3]))
                    if crop_w > 0 and crop_h > 0:
                        img2_crop = img2.crop((target2_bbox[0], target2_bbox[1], target2_bbox[0]+crop_w, target2_bbox[1]+crop_h))
                        paste_x = random.randint(0, img1_size[0]-crop_w)
                        paste_y = img1_size[1]
                        img1.paste(img2_crop, (paste_x, paste_y-crop_h, paste_x+crop_w, paste_y))
                images_blend.append(img1)
            else:
                images_blend.append(images[idx])

        return tuple(images_blend), tuple(targets)
