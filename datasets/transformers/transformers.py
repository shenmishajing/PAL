import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


def rotate_img(img, angle):
    """
    img   --image
    angle --rotation angle
    return--rotated img
    """
    h, w = img.shape[:2]
    rotate_center = (w / 2, h / 2)
    # 获取旋转矩阵
    # 参数1为旋转中心点;
    # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
    # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    # 计算图像新边界
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    # 调整旋转矩阵以考虑平移
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h))


@TRANSFORMS.register_module()
class RotateImage(BaseTransform):
    """Rotate the image by theta.

    Required Keys:

    - img
    - theta

    Modified Keys:

    - img

    """

    def transform(self, results: dict) -> dict:
        """Transform function to random shift images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        if "theta" in results:
            results["img"] = rotate_img(results["img"], results["theta"])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
