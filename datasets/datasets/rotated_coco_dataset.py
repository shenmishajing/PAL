# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools
import os
import pickle
import random
from collections import defaultdict
from typing import List, Union

from mmdet.datasets import CocoDataset
from mmengine.dataset import force_full_init


class RotatedCocoDataset(CocoDataset):
    """Rotated dataset for COCO."""

    def __init__(
        self,
        *args,
        theta: float = 5,
        max_theta: float = 45,
        add_origin: bool = True,
        length: int = None,
        **kwargs,
    ) -> None:
        self.theta = theta
        self.max_theta = max_theta
        self.add_origin = add_origin
        self.length = length
        super().__init__(*args, **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info["raw_img_info"]
        ann_info = raw_data_info["raw_ann_info"]

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = os.path.join(self.data_prefix["img"], img_info["file_name"])
        if self.data_prefix.get("seg", None):
            seg_map_path = os.path.join(
                self.data_prefix["seg"],
                img_info["file_name"].rsplit(".", 1)[0] + self.seg_map_suffix,
            )
        else:
            seg_map_path = None
        data_info["img_path"] = img_path
        data_info["img_id"] = img_info["img_id"]
        data_info["seg_map_path"] = seg_map_path
        data_info["height"] = img_info["height"]
        data_info["width"] = img_info["width"]

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get("iscrowd", False):
                instance["ignore_flag"] = 1
            else:
                instance["ignore_flag"] = 0
            instance["bbox"] = bbox
            instance["bbox_label"] = self.cat2label[ann["category_id"]]

            if ann.get("segmentation", None):
                instance["mask"] = ann["segmentation"]

            instances.append(instance)
        data_info["instances"] = instances
        return data_info

    def load_rotated_data_list(self) -> List[dict]:
        data_list = defaultdict(list)
        origin_ann_file = self.ann_file
        for theta in range(0, self.max_theta, self.theta):
            theta = int(theta + self.theta)
            ann_file = origin_ann_file.replace(
                "annotations", f"rotated_annotations/{(theta)}"
            )

            if os.path.exists(ann_file):
                self.ann_file = ann_file
            else:
                self.ann_file = origin_ann_file

            for data in super().load_data_list():
                data["theta"] = theta
                data_list[data["img_id"]].append(data)
        self.ann_file = origin_ann_file
        if self.add_origin:
            for data in super().load_data_list():
                data["theta"] = 0
                data_list[data["img_id"]].append(data)
        data_list = list(data_list.values())
        if self.length is not None:
            data_list = data_list[: self.length]
        return data_list

    def load_data_list(self) -> List[dict]:
        return list(itertools.chain.from_iterable(self.load_rotated_data_list()))

