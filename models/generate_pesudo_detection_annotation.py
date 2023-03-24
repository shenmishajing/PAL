import copy
import json
import os

from .mmdet_model_adapter import MMDetModelAdapter


class GeneratePesudeDetectionAnnotation(MMDetModelAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_paths = ["annotation"]

    def on_predict_epoch_start(self) -> None:
        self.annotation_output_path = self.trainer.datamodule.dataset.ann_file.replace(
            "annotations", "rotated_annotations/{theta}"
        )

        ann = json.load(open(self.trainer.datamodule.dataset.ann_file))
        ann.pop("annotations")

        self.ann_id = {}
        self.ann_jsons = {}
        for theta in range(
            0,
            self.trainer.datamodule.dataset.max_theta,
            self.trainer.datamodule.dataset.theta,
        ):
            theta = int(theta + self.trainer.datamodule.dataset.theta)
            self.ann_id[theta] = 0
            self.ann_jsons[theta] = copy.deepcopy(ann)
            os.makedirs(
                os.path.dirname(self.annotation_output_path.format(theta=theta)),
                exist_ok=True,
            )

    def predict_forward(self, batch, *args, **kwargs):
        return dict(predict_outputs=self(batch, mode="predict"))

    def annotation_visualization(self, *args, predict_outputs, **kwargs):
        for output in predict_outputs:
            inds = output.pred_instances.scores > 0.5
            bboxes = output.pred_instances.bboxes[inds].cpu().tolist()
            labels = output.pred_instances.labels[inds].cpu().tolist()
            for box, label in zip(bboxes, labels):
                theta = int(output.theta)
                self.ann_id[theta] += 1
                self.ann_jsons[theta].append(
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

    def on_predict_epoch_end(self) -> None:
        for theta in range(
            0,
            self.trainer.datamodule.dataset.max_theta,
            self.trainer.datamodule.dataset.theta,
        ):
            theta = int(theta + self.trainer.datamodule.dataset.theta)
            json.dump(
                self.ann_jsons[theta],
                open(self.annotation_output_path.format(theta=theta), "w"),
            )
