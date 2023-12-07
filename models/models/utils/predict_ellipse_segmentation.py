import cv2
import numpy as np
import torch


def predict_ellipse_segmentation(batch_data_samples):
    for data_sample in batch_data_samples:
        result = data_sample.pred_instances
        if result.bboxes.shape[0]:
            masks = []
            bboxes = result.bboxes.cpu()
            for bbox in bboxes:
                mask = np.zeros(
                    (data_sample.ori_shape[0], data_sample.ori_shape[1]),
                    dtype=np.uint8,
                )
                cv2.ellipse(
                    mask,
                    center=(
                        int((bbox[0] + bbox[2]) / 2),
                        int((bbox[1] + bbox[3]) / 2),
                    ),
                    axes=(
                        int((bbox[2] - bbox[0]) / 2),
                        int((bbox[3] - bbox[1]) / 2),
                    ),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=255,
                    thickness=-1,
                )
                masks.append(mask > 0)
            result.masks = torch.from_numpy(np.stack(masks, 0)).to(result.bboxes.device)
        else:
            result.masks = torch.zeros(
                (0, data_sample.ori_shape[0], data_sample.ori_shape[1]),
                dtype=torch.bool,
                device=result.bboxes.device,
            )
