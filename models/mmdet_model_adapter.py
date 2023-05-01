import os
import shutil
<<<<<<< HEAD
from abc import ABC
from types import MethodType

import cv2
import mmcv
import torch
from mmengine import MessageHub
from mmengine.model import BaseModule
from torch import nn

from .base import LightningModule


class MMDetModelAdapter(LightningModule, BaseModule, ABC):
    """Lightning module specialized for EfficientDet, with metrics support.

    The methods `forward`, `training_step`, `validation_step`, `validation_epoch_end`
    are already overriden.

    # Arguments
        model: The pytorch model to use.
        metrics: `Sequence` of metrics to use.

    # Returns
        A `LightningModule`.
    """

    def __init__(
        self,
        model: nn.Module,
        visualizer_kwargs=None,
        *args,
        **kwargs,
    ):
        """
        To show a metric in the progressbar a list of tupels can be provided for metrics_log_info, the first
        entry has to be the name of the metric to log and the second entry the display name in the progressbar. By default the
        mAP is logged to the progressbar.
        """
        super().__init__(*args, **kwargs)
        # self._output_paths = ["cam", "result"]
        self._output_paths = ["result"]

        self.model = model
        self.model.data_preprocessor.cast_data = MethodType(
            lambda obj, data: data, self.model.data_preprocessor
        )

        if visualizer_kwargs is None:
            self.visualizer_kwargs = {}
        else:
            self.visualizer_kwargs = visualizer_kwargs

    def _dump_init_info(self, *args, **kwargs):
        pass

    def on_fit_start(self):
        self.init_weights()

    def forward(self, batch, mode="loss"):
        self.batch_size = len(batch["inputs"])
        batch = self.model.data_preprocessor(batch, mode != "predict")
        return self.model._run_forward(batch, mode=mode)

    def forward_step(self, batch, *args, split="val", **kwargs):
        outputs = self(batch, mode="predict")
        self.trainer.datamodule.evaluators[split].process(outputs, batch)
        return outputs

    def on_forward_epoch_end(self, *args, split="val", **kwargs):
        log_vars = self.trainer.datamodule.evaluators[split].evaluate(
            len(self.trainer.datamodule.datasets[split])
        )
        self.log_dict(self.flatten_dict(log_vars, split), sync_dist=True)
        return log_vars

    def training_step(self, batch, *args, **kwargs):
        message_hub = MessageHub.get_current_instance()
        message_hub.update_info("iter", self.trainer.global_step)
        _, log_vars = self.model.parse_losses(self(batch))
        self.log_dict(self.flatten_dict(log_vars))
        return log_vars

    def predict_forward(self, batch, *args, **kwargs):
        predict_outputs = self(batch, mode="predict")

        for output in predict_outputs:
            # rescale gt bboxes
            assert output.get("scale_factor") is not None
            output.gt_instances.bboxes /= output.gt_instances.bboxes.new_tensor(
                output.scale_factor
            ).repeat((1, 2))

        batch = self.model.data_preprocessor(batch)

        if isinstance(batch, dict):
            inputs = batch["inputs"]
            data_samples = batch["data_samples"]
        elif isinstance(batch, (list, tuple)):
            inputs, data_samples = batch

        feature_map_outputs = [
            nn.functional.interpolate(
                output.mean(dim=1, keepdim=True),
                size=data_samples[0].batch_input_shape,
                mode="bilinear",
            )
            for output in self.model.extract_feat(inputs)
        ]
        feature_map_outputs = [
            ((output - output.min()) / (output.max() - output.min(output)) * 255 + 0.5)
            .clamp_(0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", torch.uint8)
            .numpy()
            for output in feature_map_outputs
        ]
        return {
            "predict_outputs": predict_outputs,
            "feature_map_outputs": feature_map_outputs,
            "data_samples": data_samples,
        }
=======

import cv2
import mmcv

from mmlab_lightning.model import MMLabModelAdapter


class MMDetModelAdapter(MMLabModelAdapter):
    def __init__(
        self,
        predict_tasks=None,
        *args,
        **kwargs,
    ):
        if predict_tasks is None:
            predict_tasks = ["cam", "result"]
        super().__init__(*args, predict_tasks=predict_tasks, **kwargs)
>>>>>>> mmdet_lightning/main

    def cam_visualization(self, *args, data_samples, feature_map_outputs, **kwargs):
        for i, data_sample in enumerate(data_samples):
            name = os.path.basename(data_sample.img_path)
            for layer_num, output in enumerate(feature_map_outputs):
                mmcv.imwrite(
                    cv2.applyColorMap(output[i], cv2.COLORMAP_JET),
                    os.path.join(
                        self.cam_output_path,
                        os.path.splitext(name)[0] + f"_{layer_num}.png",
                    ),
                )
            shutil.copy2(
                data_sample.img_path,
                os.path.join(self.cam_output_path, name),
            )

    def result_visualization(self, *args, predict_outputs, **kwargs):
        for output in predict_outputs:
            # result visualization
            name = os.path.basename(output.img_path)
            self.trainer.datamodule.visualizers["predict"].add_datasample(
                name,
                mmcv.imread(output.img_path, channel_order="rgb"),
                output,
                out_file=os.path.join(self.result_output_path, name),
                **self.visualizer_kwargs,
            )
