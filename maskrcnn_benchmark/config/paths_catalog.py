# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
      "coco_human_parsing": {
        "img_dir": "full_product_det/human_parsing",
        "ann_file": "full_product_det/annotations/human_parsing.json"
      },
      "coco_cat09": {
        "img_dir": "full_product_det/cat09",
        "ann_file": "full_product_det/annotations/cat09.json"
      },
      "coco_shengxian": {
        "img_dir": "full_product_det/shengxian",
        "ann_file": "full_product_det/annotations/shengxian.json"
      },
      "coco_jiadian2": {
        "img_dir": "full_product_det/jiadian2",
        "ann_file": "full_product_det/annotations/jiadian2.json"
      },
      "coco_meizhuang_1w": {
        "img_dir": "full_product_det/meizhuang_1w",
        "ann_file": "full_product_det/annotations/meizhuang_1w.json"
      },
      "coco_bag_purse_8000": {
        "img_dir": "full_product_det/bag_purse_8000",
        "ann_file": "full_product_det/annotations/bag_purse_8000.json"
      },
      "coco_jiulei": {
        "img_dir": "full_product_det/jiulei",
        "ann_file": "full_product_det/annotations/jiulei.json"
      },
      "coco_shouji_extra": {
        "img_dir": "full_product_det/shouji_extra",
        "ann_file": "full_product_det/annotations/shouji_extra.json"
      },
      "coco_shoes": {
        "img_dir": "full_product_det/shoes",
        "ann_file": "full_product_det/annotations/shoes.json"
      },
      "coco_cat06": {
        "img_dir": "full_product_det/cat06",
        "ann_file": "full_product_det/annotations/cat06.json"
      },
      "coco_shipinbuchong": {
        "img_dir": "full_product_det/shipinbuchong",
        "ann_file": "full_product_det/annotations/shipinbuchong.json"
      },
      "coco_zhubaoshoushi": {
        "img_dir": "full_product_det/zhubaoshoushi",
        "ann_file": "full_product_det/annotations/zhubaoshoushi.json"
      },
      "coco_jiadian1_extra": {
        "img_dir": "full_product_det/jiadian1_extra",
        "ann_file": "full_product_det/annotations/jiadian1_extra.json"
      },
      "coco_tushu": {
        "img_dir": "full_product_det/tushu",
        "ann_file": "full_product_det/annotations/tushu.json"
      },
      "coco_cat04": {
        "img_dir": "full_product_det/cat04",
        "ann_file": "full_product_det/annotations/cat04.json"
      },
      "coco_tushu1": {
        "img_dir": "full_product_det/tushu1",
        "ann_file": "full_product_det/annotations/tushu1.json"
      },
      "coco_shouji": {
        "img_dir": "full_product_det/shouji",
        "ann_file": "full_product_det/annotations/shouji.json"
      },
      "coco_muying_shipin": {
        "img_dir": "full_product_det/muying_shipin",
        "ann_file": "full_product_det/annotations/muying_shipin.json"
      },
      "coco_jiazhuang": {
        "img_dir": "full_product_det/jiazhuang",
        "ann_file": "full_product_det/annotations/jiazhuang.json"
      },
      "coco_car": {
        "img_dir": "full_product_det/car",
        "ann_file": "full_product_det/annotations/car.json"
      },
      "coco_bag": {
        "img_dir": "full_product_det/bag",
        "ann_file": "full_product_det/annotations/bag.json"
      },
      "coco_muying_jiaju": {
        "img_dir": "full_product_det/muying_jiaju",
        "ann_file": "full_product_det/annotations/muying_jiaju.json"
      },
      "coco_meizhuang_2w": {
        "img_dir": "full_product_det/meizhuang_2w",
        "ann_file": "full_product_det/annotations/meizhuang_2w.json"
      },
      "coco_wanjuyueqi": {
        "img_dir": "full_product_det/wanjuyueqi",
        "ann_file": "full_product_det/annotations/wanjuyueqi.json"
      },
      "coco_cat05": {
        "img_dir": "full_product_det/cat05",
        "ann_file": "full_product_det/annotations/cat05.json"
      },
      "coco_feicui": {
        "img_dir": "full_product_det/feicui",
        "ann_file": "full_product_det/annotations/feicui.json"
      },
      "coco_cat10": {
        "img_dir": "full_product_det/cat10",
        "ann_file": "full_product_det/annotations/cat10.json"
      },
      "coco_muying_wanjuyuqi": {
        "img_dir": "full_product_det/muying_wanjuyuqi",
        "ann_file": "full_product_det/annotations/muying_wanjuyuqi.json"
      },
      "coco_suitcase_2w": {
        "img_dir": "full_product_det/suitcase_2w",
        "ann_file": "full_product_det/annotations/suitcase_2w.json"
      },
      "coco_diannaobangong": {
        "img_dir": "full_product_det/diannaobangong",
        "ann_file": "full_product_det/annotations/diannaobangong.json"
      },
      "coco_jiadian1": {
        "img_dir": "full_product_det/jiadian1",
        "ann_file": "full_product_det/annotations/jiadian1.json"
      },
      "coco_ali_clothing": {
        "img_dir": "full_product_det/ali_clothing",
        "ann_file": "full_product_det/annotations/ali_clothing.json"
      },
      "coco_hetianyu": {
        "img_dir": "full_product_det/hetianyu",
        "ann_file": "full_product_det/annotations/hetianyu.json"
      },
      "coco_cat07": {
        "img_dir": "full_product_det/cat07",
        "ann_file": "full_product_det/annotations/cat07.json"
      },
      "coco_jiadian2_extra": {
        "img_dir": "full_product_det/jiadian2_extra",
        "ann_file": "full_product_det/annotations/jiadian2_extra.json"
      },
      "coco_shipinyinliao": {
        "img_dir": "full_product_det/shipinyinliao",
        "ann_file": "full_product_det/annotations/shipinyinliao.json"
      },
      "coco_yiyaobaojian": {
        "img_dir": "full_product_det/yiyaobaojian",
        "ann_file": "full_product_det/annotations/yiyaobaojian.json"
      },
      "coco_clothes_nohuman5k": {
        "img_dir": "full_product_det/clothes_nohuman5k",
        "ann_file": "full_product_det/annotations/clothes_nohuman5k.json"
      },
      "coco_jiadian3": {
        "img_dir": "full_product_det/jiadian3",
        "ann_file": "full_product_det/annotations/jiadian3.json"
      },
      "coco_shoujike_annotated": {
        "img_dir": "full_product_det/shoujike_annotated",
        "ann_file": "full_product_det/annotations/shoujike_annotated.json"
      },
      "coco_neiyiku_4k": {
        "img_dir": "full_product_det/neiyiku_4k",
        "ann_file": "full_product_det/annotations/neiyiku_4k.json"
      },
      "coco_chuju": {
        "img_dir": "full_product_det/chuju",
        "ann_file": "full_product_det/annotations/chuju.json"
      },
      "coco_cat08": {
        "img_dir": "full_product_det/cat08",
        "ann_file": "full_product_det/annotations/cat08.json"
      },
      "coco_cat11": {
        "img_dir": "full_product_det/cat11",
        "ann_file": "full_product_det/annotations/cat11.json"
      },
      "coco_jiaju": {
        "img_dir": "full_product_det/jiaju",
        "ann_file": "full_product_det/annotations/jiaju.json"
      },
      "coco_tushu2": {
        "img_dir": "full_product_det/tushu2",
        "ann_file": "full_product_det/annotations/tushu2.json"
      },
      "coco_shuma": {
        "img_dir": "full_product_det/shuma",
        "ann_file": "full_product_det/annotations/shuma.json"
      },
      "coco_tushu3": {
        "img_dir": "full_product_det/tushu3",
        "ann_file": "full_product_det/annotations/tushu3.json"
      }
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
