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


class RotatedCocoOriginAnnDataset(CocoDataset):
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

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with self.file_client.get_local_path(self.ann_file) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo["classes"])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        if self.length is not None:
            img_ids = img_ids[: self.length]
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info["img_id"] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info(
                {"raw_ann_info": raw_ann_info, "raw_img_info": raw_img_info}
            )

            if self.add_origin:
                start_theta = 0
            else:
                start_theta = self.theta
            for theta in range(start_theta, self.max_theta + self.theta, self.theta):
                cur_data_info = copy.copy(parsed_data_info)
                cur_data_info["theta"] = theta
                data_list.append(cur_data_info)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list


class RotatedCocoAllAnnDataset(RotatedCocoOriginAnnDataset):
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

            for data in super(RotatedCocoOriginAnnDataset, self).load_data_list():
                data["theta"] = theta
                data_list[data["img_id"]].append(data)
        self.ann_file = origin_ann_file
        if self.add_origin:
            for data in super(RotatedCocoOriginAnnDataset, self).load_data_list():
                data["theta"] = 0
                data_list[data["img_id"]].append(data)
        data_list = list(data_list.values())
        if self.length is not None:
            data_list = data_list[: self.length]
        return data_list

    def load_data_list(self) -> List[dict]:
        return itertools.chain.from_iterable(self.load_rotated_data_list())


class RotatedCocoAugAnnDataset(RotatedCocoAllAnnDataset):
    def load_data_list(self) -> List[dict]:
        return self.load_rotated_data_list()

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get("filter_empty_gt", False)
        min_size = self.filter_cfg.get("min_size", 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info[0]["img_id"] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info[0]["img_id"]
            width = data_info[0]["width"]
            height = data_info[0]["height"]
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        data_info = random.choice(data_info)
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info["sample_idx"] = idx
        else:
            data_info["sample_idx"] = len(self) + idx

        return data_info
