from mmdet.structures import SampleList
from torch import Tensor

from models.models.single_stage_instance_seg import SingleStageInstanceSegmentor


class RotateSingleStageInstanceSegmentor(SingleStageInstanceSegmentor):
    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, **kwargs
    ) -> dict:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = dict()

        positive_infos = None
        # CondInst and YOLACT have bbox_head
        if self.with_bbox:
            gt_instances = [sample.gt_instances for sample in batch_data_samples]
            for sample in batch_data_samples:
                sample.gt_instances = sample.gt_instances[
                    sample.gt_instances.theta == 0
                ]
            bbox_losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
            for sample, gt_instance in zip(batch_data_samples, gt_instances):
                sample.gt_instances = gt_instance
            losses.update(bbox_losses)
            # get positive information from bbox head, which will be used
            # in the following mask head.
            positive_infos = self.bbox_head.get_positive_infos()

        mask_loss = self.mask_head.loss(
            x, batch_data_samples, positive_infos=positive_infos, **kwargs
        )
        # avoid loss override
        assert not set(mask_loss.keys()) & set(losses.keys())

        losses.update(mask_loss)
        return losses
