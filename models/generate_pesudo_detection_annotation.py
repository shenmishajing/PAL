import copy
import json
import os
from collections import defaultdict

import mmcv
import torch

from datasets import (
    get_bboxes_from_points,
    prepare_bbox_points,
    reverse_rotate_points,
    rotate_img,
)

from .mmdet_model_adapter import MMDetModelAdapter


def get_boxes_from_anns(anns):
    boxes = torch.tensor([ann["bbox"] for ann in anns])
    if not len(boxes):
        return None
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def get_center_points(bboxes):
    return (bboxes[..., :2] + bboxes[..., 2:]) / 2


def get_area(bboxes):
    return (bboxes[..., 2] - bboxes[..., 0]) * (bboxes[..., 3] - bboxes[..., 1])


def get_iou(box1, box2):
    """
    box1: (N, 4) tensor, N个矩形框的坐标，每个矩形框由左上角和右下角坐标表示
    box2: (M, 4) tensor, M个矩形框的坐标，每个矩形框由左上角和右下角坐标表示
    """
    N = box1.size(0)
    M = box2.size(0)

    # 计算矩形框的面积
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # 计算交集的左上角和右下角坐标
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # (N, M, 2)
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # (N, M, 2)

    # 计算交集的面积
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # 计算并集的面积
    union = area1[:, None] + area2 - inter

    # 计算IoU
    iou = inter / union

    return iou


class GeneratePesudeDetectionAnnotation(MMDetModelAdapter):
    def __init__(self, predict_tasks=None, *args, **kwargs):
        if predict_tasks is None:
            predict_tasks = ["annotation"]
        super().__init__(*args, predict_tasks=predict_tasks, **kwargs)

    def on_predict_epoch_start(self) -> None:
        super().on_predict_epoch_start()

        if "annotation" in self.predict_tasks:
            self.predict_tasks[
                "annotation"
            ] = self.trainer.datamodule.dataset.ann_file.replace(
                "annotations", "rotated_annotations/{theta}"
            )

            ann = json.load(open(self.trainer.datamodule.dataset.ann_file))
            ann["annotations"] = []

            self.ann_id = {}
            self.ann_jsons = {}
            self.rotated_anns = {}
            for theta in range(
                0,
                self.trainer.datamodule.dataset.max_theta
                + self.trainer.datamodule.dataset.theta,
                self.trainer.datamodule.dataset.theta,
            ):
                theta = int(theta)
                self.ann_id[theta] = 1
                self.ann_jsons[theta] = copy.deepcopy(ann)
                self.rotated_anns[theta] = defaultdict(list)

    def predict_annotation(self, *args, predict_outputs, **kwargs):
        for output in predict_outputs:
            pred_instances = output.pred_instances
            pred_instances = pred_instances[
                pred_instances.scores
                > self.visualizer_kwargs.get("pred_score_thr", 0.3)
            ]
            bboxes = pred_instances.bboxes.cpu().tolist()
            labels = pred_instances.labels.cpu().tolist()
            for box, label in zip(bboxes, labels):
                theta = int(output.theta)
                ann = {
                    "image_id": output.img_id,
                    "category_id": self.trainer.datamodule.dataset.cat_ids[label],
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    "area": (box[2] - box[0]) * (box[3] - box[1]),
                    "iscrowd": 0,
                    "segmentation": [],
                    "id": self.ann_id[theta],
                    "ignore": 0,
                }
                self.ann_id[theta] += 1
                self.ann_jsons[theta]["annotations"].append(ann)
                self.rotated_anns[theta][(ann["image_id"], ann["category_id"])].append(
                    ann
                )

    def predict_result(self, *args, predict_outputs, output_path, **kwargs):
        for output in predict_outputs:
            # result visualization
            name = os.path.basename(output.img_path)
            image = mmcv.imread(output.img_path, channel_order="rgb")
            if output.get("theta"):
                name, ext = os.path.splitext(name)
                name = f"{name}_theta={int(output.theta)}{ext}"
                image = rotate_img(image, output.theta)
            self.trainer.datamodule.visualizers["predict"].add_datasample(
                name,
                image,
                output,
                out_file=os.path.join(output_path, name),
                draw_gt=False,
                **self.visualizer_kwargs,
            )

    def predict_rotate_back_result(self, *args, predict_outputs, output_path, **kwargs):
        for output in predict_outputs:
            # result visualization
            output.pred_instances.bboxes = get_bboxes_from_points(
                reverse_rotate_points(
                    prepare_bbox_points(output.pred_instances.bboxes),
                    output.theta,
                    output.ori_shape,
                )
            )

            name = os.path.basename(output.img_path)
            image = mmcv.imread(output.img_path, channel_order="rgb")
            if output.get("theta"):
                name, ext = os.path.splitext(name)
                name = f"{name}_theta={int(output.theta)}{ext}"
            self.trainer.datamodule.visualizers["predict"].add_datasample(
                name,
                image,
                output,
                out_file=os.path.join(output_path, name),
                **self.visualizer_kwargs,
            )

    def on_predict_epoch_end(self) -> None:
        if "annotation" in self.predict_tasks:
            # get associated rotate annotations
            image_shape = {
                ann["id"]: (ann["height"], ann["width"])
                for ann in self.ann_jsons[0]["images"]
            }
            image_ids = set(
                [ann["image_id"] for ann in self.ann_jsons[0]["annotations"]]
            )
            category_ids = set(
                [ann["category_id"] for ann in self.ann_jsons[0]["annotations"]]
            )
            self.ann_jsons[0]["rotated_annotations"] = defaultdict(list)

            for image_id in image_ids:
                for category_id in category_ids:
                    bboxes = get_boxes_from_anns(
                        self.rotated_anns[0][(image_id, category_id)]
                    )
                    if bboxes is None:
                        continue
                    center_points = (bboxes[..., :2] + bboxes[..., 2:]) / 2
                    area = (bboxes[..., 2] - bboxes[..., 0]) * (
                        bboxes[..., 3] - bboxes[..., 1]
                    )
                    for theta in range(
                        0,
                        self.trainer.datamodule.dataset.max_theta,
                        self.trainer.datamodule.dataset.theta,
                    ):
                        theta = int(theta + self.trainer.datamodule.dataset.theta)
                        cur_bboxes = get_boxes_from_anns(
                            self.rotated_anns[theta][(image_id, category_id)]
                        )
                        if cur_bboxes is None:
                            continue
                        cur_reversed_bboxes = reverse_rotate_points(
                            prepare_bbox_points(cur_bboxes),
                            theta,
                            image_shape[image_id],
                        )
                        cur_big_boxes = get_bboxes_from_points(cur_reversed_bboxes)
                        cur_iou = get_iou(bboxes, cur_big_boxes)

                        count = (cur_iou > 0.1).sum(dim=-1)
                        iou_inds = cur_iou.max(dim=-1)[1]
                        inds = iou_inds.new_full((bboxes.shape[0],), -1)
                        inds[count == 1] = iou_inds[count == 1]

                        if (count > 1).any():
                            cur_area = (cur_bboxes[..., 2] - cur_bboxes[..., 0]) * (
                                cur_bboxes[..., 3] - cur_bboxes[..., 1]
                            )
                            cur_center_points = cur_reversed_bboxes.mean(dim=-2)
                            center_point_loss = (
                                (cur_center_points[None] - center_points[:, None])
                                .pow(2)
                                .sum(dim=-1)
                                / area[:, None]
                            ).sqrt()
                            area_loss = (
                                (cur_area[None] - area[:, None]).abs() / area[:, None]
                            ).sqrt()
                            loss = center_point_loss + area_loss
                            min_loss, min_loss_idx = loss.min(dim=-1)
                            inds[count > 1] = min_loss_idx[count > 1]

                        inds = inds.cpu().numpy().tolist()

                        for ann, idx in zip(
                            self.rotated_anns[0][(image_id, category_id)], inds
                        ):
                            if idx == -1:
                                continue
                            self.ann_jsons[0]["rotated_annotations"][ann["id"]].append(
                                [
                                    theta,
                                    self.rotated_anns[theta][(image_id, category_id)][
                                        idx
                                    ]["bbox"],
                                ]
                            )

            # save for every theta rotated annotations
            for theta in range(
                0,
                self.trainer.datamodule.dataset.max_theta
                + self.trainer.datamodule.dataset.theta,
                self.trainer.datamodule.dataset.theta,
            ):
                theta = int(theta)
                os.makedirs(
                    os.path.dirname(
                        self.predict_tasks["annotation"].format(theta=theta)
                    ),
                    exist_ok=True,
                )
                json.dump(
                    self.ann_jsons[theta],
                    open(self.predict_tasks["annotation"].format(theta=theta), "w"),
                )
