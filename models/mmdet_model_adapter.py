import os
import shutil

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

    def predict_forward(self, batch, *args, **kwargs):
        predict_result = super().predict_forward(batch, *args, **kwargs)
        for output in predict_result["predict_outputs"]:
            # rescale gt bboxes
            assert output.get("scale_factor") is not None
            output.gt_instances.bboxes /= output.gt_instances.bboxes.new_tensor(
                output.scale_factor
            ).repeat((1, 2))
        return predict_result

    def predict_cam(
        self, *args, data_samples, feature_map_outputs, output_path, **kwargs
    ):
        for i, data_sample in enumerate(data_samples):
            name = os.path.basename(data_sample.img_path)
            for layer_num, output in enumerate(feature_map_outputs):
                mmcv.imwrite(
                    cv2.applyColorMap(output[i], cv2.COLORMAP_JET),
                    os.path.join(
                        output_path,
                        os.path.splitext(name)[0] + f"_{layer_num}.png",
                    ),
                )
            shutil.copy2(
                data_sample.img_path,
                os.path.join(output_path, name),
            )

    def predict_result(self, *args, predict_outputs, output_path, **kwargs):
        for output in predict_outputs:
            # result visualization
            name = os.path.basename(output.img_path)
            self.trainer.datamodule.visualizers["predict"].add_datasample(
                name,
                mmcv.imread(output.img_path, channel_order="rgb"),
                output,
                out_file=os.path.join(output_path, name),
                **self.visualizer_kwargs,
            )
