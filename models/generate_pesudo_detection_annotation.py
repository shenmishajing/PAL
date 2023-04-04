import copy
import json
import os

import mmcv

from datasets import rotate_img

from .mmdet_model_adapter import MMDetModelAdapter


class GeneratePesudeDetectionAnnotation(MMDetModelAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_paths = ["annotation"]

    def on_predict_epoch_start(self) -> None:
        super().on_predict_epoch_start()

        self.annotation_output_path = self.trainer.datamodule.dataset.ann_file.replace(
            "annotations", "rotated_annotations/{theta}"
        )

        ann = json.load(open(self.trainer.datamodule.dataset.ann_file))
        ann["annotations"] = []

        self.ann_id = {}
        self.ann_jsons = {}
        for theta in range(
            0,
            self.trainer.datamodule.dataset.max_theta
            + self.trainer.datamodule.dataset.theta,
            self.trainer.datamodule.dataset.theta,
        ):
            theta = int(theta)
            self.ann_id[theta] = 0
            self.ann_jsons[theta] = copy.deepcopy(ann)

    def predict_forward(self, batch, *args, **kwargs):
        return dict(predict_outputs=self(batch, mode="predict"))

    def annotation_visualization(self, *args, predict_outputs, **kwargs):
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
                self.ann_id[theta] += 1
                self.ann_jsons[theta]["annotations"].append(
                    {
                        "image_id": output.img_id,
                        "category_id": self.trainer.datamodule.dataset.cat_ids[label],
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "area": (box[2] - box[0]) * (box[3] - box[1]),
                        "iscrowd": 0,
                        "segmentation": [],
                        "id": self.ann_id[theta],
                        "ignore": 0,
                    }
                )

    def result_visualization(self, *args, predict_outputs, **kwargs):
        for output in predict_outputs:
            # rescale gt bboxes
            assert output.get("scale_factor") is not None
            output.gt_instances.bboxes /= output.gt_instances.bboxes.new_tensor(
                output.scale_factor
            ).repeat((1, 2))

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
                out_file=os.path.join(self.result_output_path, name),
                **self.visualizer_kwargs,
            )

    def on_predict_epoch_end(self) -> None:
        for theta in range(
            0,
            self.trainer.datamodule.dataset.max_theta
            + self.trainer.datamodule.dataset.theta,
            self.trainer.datamodule.dataset.theta,
        ):
            theta = int(theta)
            os.makedirs(
                os.path.dirname(self.annotation_output_path.format(theta=theta)),
                exist_ok=True,
            )
            json.dump(
                self.ann_jsons[theta],
                open(self.annotation_output_path.format(theta=theta), "w"),
            )
