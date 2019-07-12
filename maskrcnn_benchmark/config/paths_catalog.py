# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
      "coco_human_parsing": {
        "img_dir": "human_parsing",
        "ann_file": "annotations/human_parsing.json"
      },
      "coco_cat09": {
        "img_dir": "cat09",
        "ann_file": "annotations/cat09.json"
      },
      "coco_shengxian": {
        "img_dir": "shengxian",
        "ann_file": "annotations/shengxian.json"
      },
      "coco_jiadian2": {
        "img_dir": "jiadian2",
        "ann_file": "annotations/jiadian2.json"
      },
      "coco_meizhuang_1w": {
        "img_dir": "meizhuang_1w",
        "ann_file": "annotations/meizhuang_1w.json"
      },
      "coco_bag_purse_8000": {
        "img_dir": "bag_purse_8000",
        "ann_file": "annotations/bag_purse_8000.json"
      },
      "coco_jiulei": {
        "img_dir": "jiulei",
        "ann_file": "annotations/jiulei.json"
      },
      "coco_shouji_extra": {
        "img_dir": "shouji_extra",
        "ann_file": "annotations/shouji_extra.json"
      },
      "coco_shoes": {
        "img_dir": "shoes",
        "ann_file": "annotations/shoes.json"
      },
      "coco_cat06": {
        "img_dir": "cat06",
        "ann_file": "annotations/cat06.json"
      },
      "coco_shipinbuchong": {
        "img_dir": "shipinbuchong",
        "ann_file": "annotations/shipinbuchong.json"
      },
      "coco_zhubaoshoushi": {
        "img_dir": "zhubaoshoushi",
        "ann_file": "annotations/zhubaoshoushi.json"
      },
      "coco_jiadian1_extra": {
        "img_dir": "jiadian1_extra",
        "ann_file": "annotations/jiadian1_extra.json"
      },
      "coco_tushu": {
        "img_dir": "tushu",
        "ann_file": "annotations/tushu.json"
      },
      "coco_cat04": {
        "img_dir": "cat04",
        "ann_file": "annotations/cat04.json"
      },
      "coco_tushu1": {
        "img_dir": "tushu1",
        "ann_file": "annotations/tushu1.json"
      },
      "coco_shouji": {
        "img_dir": "shouji",
        "ann_file": "annotations/shouji.json"
      },
      "coco_muying_shipin": {
        "img_dir": "muying_shipin",
        "ann_file": "annotations/muying_shipin.json"
      },
      "coco_jiazhuang": {
        "img_dir": "jiazhuang",
        "ann_file": "annotations/jiazhuang.json"
      },
      "coco_car": {
        "img_dir": "car",
        "ann_file": "annotations/car.json"
      },
      "coco_bag": {
        "img_dir": "bag",
        "ann_file": "annotations/bag.json"
      },
      "coco_muying_jiaju": {
        "img_dir": "muying_jiaju",
        "ann_file": "annotations/muying_jiaju.json"
      },
      "coco_meizhuang_2w": {
        "img_dir": "meizhuang_2w",
        "ann_file": "annotations/meizhuang_2w.json"
      },
      "coco_meizhuang_2w_v3": {
        "img_dir": "meizhuang_2w",
        "ann_file": "annotations/meizhuang_2w_v3.json"
      },
      "coco_wanjuyueqi": {
        "img_dir": "wanjuyueqi",
        "ann_file": "annotations/wanjuyueqi.json"
      },
      "coco_cat05": {
        "img_dir": "cat05",
        "ann_file": "annotations/cat05.json"
      },
      "coco_feicui": {
        "img_dir": "feicui",
        "ann_file": "annotations/feicui.json"
      },
      "coco_cat10": {
        "img_dir": "cat10",
        "ann_file": "annotations/cat10.json"
      },
      "coco_muying_wanjuyuqi": {
        "img_dir": "muying_wanjuyuqi",
        "ann_file": "annotations/muying_wanjuyuqi.json"
      },
      "coco_suitcase_2w": {
        "img_dir": "suitcase_2w",
        "ann_file": "annotations/suitcase_2w.json"
      },
      "coco_diannaobangong": {
        "img_dir": "diannaobangong",
        "ann_file": "annotations/diannaobangong.json"
      },
      "coco_jiadian1": {
        "img_dir": "jiadian1",
        "ann_file": "annotations/jiadian1.json"
      },
      "coco_ali_clothing": {
        "img_dir": "ali_clothing",
        "ann_file": "annotations/ali_clothing.json"
      },
      "coco_hetianyu": {
        "img_dir": "hetianyu",
        "ann_file": "annotations/hetianyu.json"
      },
      "coco_cat07": {
        "img_dir": "cat07",
        "ann_file": "annotations/cat07.json"
      },
      "coco_jiadian2_extra": {
        "img_dir": "jiadian2_extra",
        "ann_file": "annotations/jiadian2_extra.json"
      },
      "coco_shipinyinliao": {
        "img_dir": "shipinyinliao",
        "ann_file": "annotations/shipinyinliao.json"
      },
      "coco_yiyaobaojian": {
        "img_dir": "yiyaobaojian",
        "ann_file": "annotations/yiyaobaojian.json"
      },
      "coco_clothes_nohuman5k": {
        "img_dir": "clothes_nohuman5k",
        "ann_file": "annotations/clothes_nohuman5k.json"
      },
      "coco_jiadian3": {
        "img_dir": "jiadian3",
        "ann_file": "annotations/jiadian3.json"
      },
      "coco_shoujike_annotated": {
        "img_dir": "shoujike_annotated",
        "ann_file": "annotations/shoujike_annotated.json"
      },
      "coco_neiyiku_4k": {
        "img_dir": "neiyiku_4k",
        "ann_file": "annotations/neiyiku_4k.json"
      },
      "coco_chuju": {
        "img_dir": "chuju",
        "ann_file": "annotations/chuju.json"
      },
      "coco_cat08": {
        "img_dir": "cat08",
        "ann_file": "annotations/cat08.json"
      },
      "coco_cat11": {
        "img_dir": "cat11",
        "ann_file": "annotations/cat11.json"
      },
      "coco_jiaju": {
        "img_dir": "jiaju",
        "ann_file": "annotations/jiaju.json"
      },
      "coco_tushu2": {
        "img_dir": "tushu2",
        "ann_file": "annotations/tushu2.json"
      },
      "coco_shuma": {
        "img_dir": "shuma",
        "ann_file": "annotations/shuma.json"
      },
      "coco_tushu3": {
        "img_dir": "tushu3",
        "ann_file": "annotations/tushu3.json"
      },
      "coco_test_det": {
        "img_dir": "test_det",
        "ann_file": "annotations/test_det.json"
      }
    }

    @staticmethod
    def get(name, dataset_prefix):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, dataset_prefix, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, dataset_prefix, attrs["ann_file"]),
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
