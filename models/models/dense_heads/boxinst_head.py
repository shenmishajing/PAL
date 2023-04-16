# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
from mmdet.models.dense_heads.boxinst_head import BoxInstMaskHead
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList
from mmengine import MessageHub
from mmengine.structures import InstanceData
from torch import Tensor

from datasets.transformers.transformers import torch_rotate_img


@MODELS.register_module()
class RotateBoxInstMaskHead(BoxInstMaskHead):
    def loss(
        self,
        x: Union[List[Tensor], Tuple[Tensor]],
        batch_data_samples: SampleList,
        positive_infos: OptInstanceList = None,
        **kwargs
    ) -> dict:
        """Perform forward propagation and loss calculation of the mask head on
        the features of the upstream network.

        Args:
            x (list[Tensor] | tuple[Tensor]): Features from FPN.
                Each has a shape (B, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            positive_infos (list[:obj:`InstanceData`], optional): Information
                of positive samples. Used when the label assignment is
                done outside the MaskHead, e.g., BboxHead in
                YOLACT or CondInst, etc. When the label assignment is done in
                MaskHead, it would be None, like SOLO or SOLOv2. All values
                in it should have shape (num_positive_samples, *).


        Returns:
            dict: A dictionary of loss components.
        """
        if positive_infos is None:
            outs = self(x)
        else:
            outs = self(x, positive_infos)

        assert isinstance(outs, tuple), (
            "Forward results should be a tuple, " "even if only one item is returned"
        )

        (
            batch_gt_instances,
            batch_gt_instances_ignore,
            batch_img_metas,
        ) = unpack_gt_instances(batch_data_samples)
        losses = self.loss_by_feat(
            *outs,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            positive_infos=positive_infos,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            **kwargs
        )
        return losses

    def loss_by_feat(
        self,
        mask_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        positive_infos: InstanceList,
        **kwargs
    ) -> dict:
        """Calculate the loss based on the features extracted by the mask head.

        Args:
            mask_preds (list[Tensor]): List of predicted masks, each has
                shape (num_classes, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``masks``,
                and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of multiple images.
            positive_infos (List[:obj:``InstanceData``]): Information of
                positive samples of each image that are assigned in detection
                head.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert (
            positive_infos is not None
        ), "positive_infos should not be None in `BoxInstMaskHead`"
        losses = dict()

        loss_mask_project = 0.0
        loss_mask_pairwise = 0.0
        num_imgs = len(mask_preds)
        total_pos = 0.0
        avg_fatcor = 0.0

        for idx in range(num_imgs):
            (
                mask_pred,
                pos_mask_targets,
                pos_pairwise_masks,
                num_pos,
            ) = self._get_targets_single(
                mask_preds[idx], batch_gt_instances[idx], positive_infos[idx]
            )
            # mask loss
            total_pos += num_pos
            if num_pos == 0 or pos_mask_targets is None:
                loss_project = mask_pred.new_zeros(1).mean()
                loss_pairwise = mask_pred.new_zeros(1).mean()
                avg_fatcor += 0.0
            else:
                loss_project = []
                for mask_p, pos_mask_target in zip(mask_pred, pos_mask_targets):
                    # compute the project term
                    loss_project_x = self.loss_mask(
                        mask_p.max(dim=1, keepdim=True)[0],
                        pos_mask_target[0],
                        reduction_override="none",
                    ).sum()
                    loss_project_y = self.loss_mask(
                        mask_p.max(dim=2, keepdim=True)[0],
                        pos_mask_target[1],
                        reduction_override="none",
                    ).sum()
                    loss_project.append(loss_project_x + loss_project_y)
                loss_project = torch.stack(loss_project).mean()
                # compute the pairwise term
                pairwise_affinity = self.get_pairwise_affinity(mask_pred[0])
                avg_fatcor += pos_pairwise_masks.sum().clamp(min=1.0)
                loss_pairwise = (pairwise_affinity * pos_pairwise_masks).sum()

            loss_mask_project += loss_project
            loss_mask_pairwise += loss_pairwise

        if total_pos == 0:
            total_pos += 1  # avoid nan
        if avg_fatcor == 0:
            avg_fatcor += 1  # avoid nan
        loss_mask_project = loss_mask_project / total_pos
        loss_mask_pairwise = loss_mask_pairwise / avg_fatcor
        message_hub = MessageHub.get_current_instance()
        iter = message_hub.get_info("iter")
        warmup_factor = min(iter / float(self.warmup_iters), 1.0)
        loss_mask_pairwise *= warmup_factor

        losses.update(
            loss_mask_project=loss_mask_project, loss_mask_pairwise=loss_mask_pairwise
        )
        return losses

    def _get_targets_single(
        self,
        mask_preds: Tensor,
        gt_instances: InstanceData,
        positive_info: InstanceData,
    ):
        """Compute targets for predictions of single image.

        Args:
            mask_preds (Tensor): Predicted prototypes with shape
                (num_classes, H, W).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes``, ``labels``,
                and ``masks`` attributes.
            positive_info (:obj:`InstanceData`): Information of positive
                samples that are assigned in detection head. It usually
                contains following keys.

                    - pos_assigned_gt_inds (Tensor): Assigner GT indexes of
                      positive proposals, has shape (num_pos, )
                    - pos_inds (Tensor): Positive index of image, has
                      shape (num_pos, ).
                    - param_pred (Tensor): Positive param preditions
                      with shape (num_pos, num_params).

        Returns:
            tuple: Usually returns a tuple containing learning targets.

            - mask_preds (Tensor): Positive predicted mask with shape
              (num_pos, mask_h, mask_w).
            - pos_mask_targets (Tensor): Positive mask targets with shape
              (num_pos, mask_h, mask_w).
            - pos_pairwise_masks (Tensor): Positive pairwise masks with
              shape: (num_pos, num_neighborhood, mask_h, mask_w).
            - num_pos (int): Positive numbers.
        """
        gt_bboxes = gt_instances.bboxes
        device = gt_bboxes.device
        # Note that pairwise_masks are generated by image color similarity
        # from BoxInstDataPreprocessor
        pairwise_masks = gt_instances.pairwise_masks
        pairwise_masks = pairwise_masks.to(device=device)

        # process with mask targets
        pos_assigned_gt_inds = positive_info.get("pos_assigned_gt_inds")
        scores = positive_info.get("scores")
        centernesses = positive_info.get("centernesses")
        num_pos = pos_assigned_gt_inds.size(0)

        if gt_bboxes.size(0) == 0 or num_pos == 0:
            return mask_preds, None, None, 0
        # Since we're producing (near) full image masks,
        # it'd take too much vram to backprop on every single mask.
        # Thus we select only a subset.
        if (self.max_masks_to_train != -1) and (num_pos > self.max_masks_to_train):
            perm = torch.randperm(num_pos)
            select = perm[: self.max_masks_to_train]
            mask_preds = mask_preds[select]
            pos_assigned_gt_inds = pos_assigned_gt_inds[select]
            num_pos = self.max_masks_to_train
        elif self.topk_masks_per_img != -1:
            unique_gt_inds = pos_assigned_gt_inds.unique()
            num_inst_per_gt = max(int(self.topk_masks_per_img / len(unique_gt_inds)), 1)

            keep_mask_preds = []
            keep_pos_assigned_gt_inds = []
            for gt_ind in unique_gt_inds:
                per_inst_pos_inds = pos_assigned_gt_inds == gt_ind
                mask_preds_per_inst = mask_preds[per_inst_pos_inds]
                gt_inds_per_inst = pos_assigned_gt_inds[per_inst_pos_inds]
                if sum(per_inst_pos_inds) > num_inst_per_gt:
                    per_inst_scores = scores[per_inst_pos_inds].sigmoid().max(dim=1)[0]
                    per_inst_centerness = (
                        centernesses[per_inst_pos_inds]
                        .sigmoid()
                        .reshape(
                            -1,
                        )
                    )
                    select = (per_inst_scores * per_inst_centerness).topk(
                        k=num_inst_per_gt, dim=0
                    )[1]
                    mask_preds_per_inst = mask_preds_per_inst[select]
                    gt_inds_per_inst = gt_inds_per_inst[select]
                keep_mask_preds.append(mask_preds_per_inst)
                keep_pos_assigned_gt_inds.append(gt_inds_per_inst)
            mask_preds = torch.cat(keep_mask_preds)
            pos_assigned_gt_inds = torch.cat(keep_pos_assigned_gt_inds)
            num_pos = pos_assigned_gt_inds.size(0)

        mask_preds_list = []
        pos_mask_targets_list = []
        for theta in sorted(
            gt_instances.theta.unique().cpu().numpy().tolist(), key=lambda x: abs(x)
        ):
            if theta != 0:
                cur_mask_preds = torch_rotate_img(mask_preds[None], theta)[0]
            else:
                cur_mask_preds = mask_preds
            cur_preds_inds = mask_preds.new_ones(mask_preds.shape[0]).bool()
            cur_theta_inds = gt_instances.theta == theta
            cur_mask_targets = [[], []]
            for i, gt_id in enumerate(pos_assigned_gt_inds):
                cur_inds = (gt_instances.id == gt_id) & cur_theta_inds
                if cur_inds.sum() < 1:
                    cur_preds_inds[i] = False
                    continue
                assert cur_inds.sum() == 1
                cur_bbox = gt_bboxes[cur_inds][0]
                pos_mask_target_x = torch.zeros(
                    (1, cur_mask_preds.shape[-1]), device=device, dtype=torch.float
                )
                pos_mask_target_x[
                    int(cur_bbox[0] / self.mask_out_stride) : int(
                        cur_bbox[2] / self.mask_out_stride
                    )
                ] = 1.0
                pos_mask_target_y = torch.zeros(
                    (cur_mask_preds.shape[-2], 1), device=device, dtype=torch.float
                )
                pos_mask_target_y[
                    int(cur_bbox[1] / self.mask_out_stride) : int(
                        cur_bbox[3] / self.mask_out_stride + 1
                    )
                ] = 1.0
                cur_mask_targets[0].append(pos_mask_target_x)
                cur_mask_targets[1].append(pos_mask_target_y)

            mask_preds_list.append(cur_mask_preds[cur_preds_inds])
            pos_mask_targets_list.append(
                [torch.stack(t, dim=0) for t in cur_mask_targets]
            )
            if theta == 0:
                pos_mask_targets = (
                    pos_mask_targets_list[-1][0] * pos_mask_targets_list[-1][1]
                )

        pos_pairwise_masks = pairwise_masks[pos_assigned_gt_inds]
        pos_pairwise_masks = pos_pairwise_masks * pos_mask_targets.unsqueeze(1)

        return (mask_preds_list, pos_mask_targets_list, pos_pairwise_masks, num_pos)
