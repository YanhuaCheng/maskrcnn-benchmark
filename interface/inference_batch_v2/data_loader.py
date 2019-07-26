# from torchvision import transforms
import os

import numpy as np
from PIL import Image, ImageFile

import transforms as T
from torch.utils.data import DataLoader, Dataset
from maskrcnn_benchmark.structures.image_list import to_image_list

ImageFile.LOAD_TRUNCATED_IMAGES = True

class dataset(Dataset):

    """load dataset."""

    def __init__(self, img_list, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.img_list[idx])
        image = Image.open(img_name).convert('RGB')
        img_h, img_w = np.array(image).shape[:2]
        if self.transform is not None:
            image = self.transform(image)

        return self.img_list[idx], image, img_h, img_w

def build_transforms(cfg):
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform

class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images_name = transposed_batch[0]
        images = transposed_batch[1]
        images_h = transposed_batch[2]
        images_w = transposed_batch[3]

        images = to_image_list(images, self.size_divisible)
        return images_name, images, images_h, images_w

def data_loader(cfg, img_names, root_dir):
    transform = build_transforms(cfg)
    img_test = dataset(img_list=img_names, root_dir=root_dir, transform=transform)
    collator = BatchCollator(32)
    img_loader = DataLoader(img_test, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, collate_fn=collator, num_workers=cfg.DATALOADER.NUM_WORKERS)

    return img_loader
