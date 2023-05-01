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
