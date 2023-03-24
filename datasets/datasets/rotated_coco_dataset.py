# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List

from mmdet.datasets import CocoDataset


class RotatedCocoDataset(CocoDataset):
    """Rotated dataset for COCO."""

    def __init__(
        self,
        *args,
        theta: float = 5,
        max_theta: float = 45,
        add_origin: bool = True,
        **kwargs,
    ) -> None:
        self.theta = theta
        self.max_theta = max_theta
        self.add_origin = add_origin
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        if self.add_origin:
            data_list = super().load_data_list()
        else:
            data_list = []

        origin_ann_file = self.ann_file
        for theta in range(0, self.max_theta, self.theta):
            ann_file = origin_ann_file.replace(
                "annotations", f"rotated_annotations/{int(theta+self.theta)}"
            )
            if os.path.exists(ann_file):
                self.ann_file = ann_file

            cur_data_list = super().load_data_list()
            for d in cur_data_list:
                d["theta"] = theta + self.theta
            data_list.extend(cur_data_list)
        self.ann_file = origin_ann_file
        return data_list
