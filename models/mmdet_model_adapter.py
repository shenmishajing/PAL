import os

import mmcv
from mmlab_lightning.models import MMLabModelAdapter


class MMDetModelAdapter(MMLabModelAdapter):
    def __init__(self, predict_tasks=None, *args, **kwargs):
        if predict_tasks is None:
            predict_tasks = ["result"]
        super().__init__(*args, predict_tasks=predict_tasks, **kwargs)

    def predict_forward(self, batch, *args, **kwargs):
        predict_result = super().predict_forward(batch, *args, **kwargs)
        for output in predict_result["predict_outputs"]:
            if output.get("gt_instances") is not None:
                # rescale gt bboxes
                if (
                    output.get("scale_factor") is not None
                    and output.gt_instances.get("bboxes") is not None
                ):
                    output.gt_instances.bboxes /= output.gt_instances.bboxes.new_tensor(
                        output.scale_factor
                    ).repeat((1, 2))

                # rescale gt masks
                if (
                    output.get("ori_shape") is not None
                    and output.get("img_shape") is not None
                    and output.img_shape != output.ori_shape
                    and output.gt_instances.get("masks") is not None
                ):
                    output.gt_instances.masks = output.gt_instances.masks.resize(
                        output.ori_shape
                    )
        return predict_result

    def predict_result(self, *args, predict_outputs, output_path, **kwargs):
        for output in predict_outputs:
            # result visualization
            name = os.path.basename(output.img_path)
            if not self.visualizer_kwargs.get("draw_gt", True):
                name = f"_{output_path.split('/')[-4]}".join(os.path.splitext(name))
            elif not self.visualizer_kwargs.get("draw_pred", True):
                name = "_ground_truth".join(os.path.splitext(name))

            self.trainer.datamodule.visualizers["predict"].add_datasample(
                name,
                mmcv.imread(output.img_path, channel_order="rgb"),
                output,
                out_file=os.path.join(output_path, name),
                **self.visualizer_kwargs,
            )
