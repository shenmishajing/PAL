import os
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import RandomFlip as _RandomFlip
from mmdet.registry import TRANSFORMS


def get_3x3_rotate_matrix(m):
    return np.concatenate([m, np.array([[0, 0, 1]])], axis=0)


def get_rotate_marix(angle, scale):
    h, w = scale
    rotate_center = (w / 2, h / 2)
    # 获取旋转矩阵
    # 参数1为旋转中心点;
    # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
    # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    # 计算图像新边界
    big_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    big_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    # 调整旋转矩阵以考虑平移
    M[0, 2] += (big_w - w) / 2
    M[1, 2] += (big_h - h) / 2
    return M, (big_h, big_w)


def get_reverse_rotate_marix(angle, scale):
    M, big_scale = get_rotate_marix(angle, scale)
    return np.linalg.pinv(get_3x3_rotate_matrix(M))[:2], big_scale


def rotate_matrix_to_theta(M_inv, new_scale, img):
    h, w = img.shape[-2:]
    new_h, new_w = new_scale
    M_inv = get_3x3_rotate_matrix(M_inv)
    return (
        img.new_tensor([[2 / w, 0, -1], [0, 2 / h, -1]])
        .mm(img.new_tensor(M_inv))
        .mm(
            img.new_tensor(
                [[2 / new_w, 0, -1], [0, 2 / new_h, -1], [0, 0, 1]]
            ).pinverse()
        )[None]
    )


def get_rotate_theta(angle, img):
    M_inv, big_scale = get_reverse_rotate_marix(angle, img.shape[-2:])
    return rotate_matrix_to_theta(M_inv, big_scale, img), big_scale


def get_reverse_rotate_theta(angle, scale, img):
    M_inv, _ = get_rotate_marix(angle, scale)
    return rotate_matrix_to_theta(M_inv, scale, img), scale


def troch_grid_sample(img, theta, scale):
    return F.grid_sample(
        img,
        F.affine_grid(theta, img.shape[:-2] + scale, align_corners=True),
        align_corners=True,
        padding_mode="reflection",
    )


def rotate_img(img, angle):
    """
    img   --image
    angle --rotation angle
    return--rotated img
    """
    M, big_scale = get_rotate_marix(angle, img.shape[:2])
    return cv2.warpAffine(
        img, M, (big_scale[1], big_scale[0]), borderMode=cv2.BORDER_REFLECT101
    )


def reverse_rotate_img(img, angle, scale):
    """
    img   --image
    angle --rotation angle
    return--rotated img
    """
    M, _ = get_reverse_rotate_marix(angle, scale)
    return cv2.warpAffine(
        img, M, (scale[1], scale[0]), borderMode=cv2.BORDER_REFLECT101
    )


def torch_rotate_img(img, angle):
    """
    img   --image
    angle --rotation angle
    return--rotated img
    """
    return troch_grid_sample(img, *get_rotate_theta(angle, img))


def torch_reverse_rotate_img(img, angle, old_scale):
    """
    img   --image
    angle --rotation angle
    return--rotated img
    """
    return troch_grid_sample(img, *get_reverse_rotate_theta(angle, old_scale, img))


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
        if "theta" in results and results["theta"]:
            results["img"] = rotate_img(results["img"], results["theta"])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


def main():
    output_path = "rotate_image"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    image_path = "data/coco_dna/train/JPEGImages_other/other-Annotated20181206XIMEACervix-10_10 (2).jpg"

    shutil.copy2(image_path, os.path.join(output_path, "origin.jpg"))

    image = cv2.imread(image_path)
    cv2.imwrite(os.path.join(output_path, "write.jpg"), image)

    for theta in range(0, 50, 5):
        rotated_image = rotate_img(image, theta)
        cv2.imwrite(os.path.join(output_path, f"rotate_{theta}.jpg"), rotated_image)

        back_image = reverse_rotate_img(rotated_image, theta, image.shape[:2])
        cv2.imwrite(os.path.join(output_path, f"back_{theta}.jpg"), back_image)

        cv2.imwrite(
            os.path.join(output_path, f"diff_{theta}.jpg"), np.abs(image - back_image)
        )

        torch_rotated_image = (
            torch_rotate_img(
                torch.from_numpy(image).permute(2, 0, 1).float().cuda()[None], theta
            )
            .to(torch.uint8)[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        cv2.imwrite(
            os.path.join(output_path, f"torch_rotate_{theta}.jpg"), torch_rotated_image
        )

        torch_back_image = (
            torch_reverse_rotate_img(
                torch.from_numpy(torch_rotated_image)
                .permute(2, 0, 1)
                .float()
                .cuda()[None],
                theta,
                image.shape[:2],
            )
            .to(torch.uint8)[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        cv2.imwrite(
            os.path.join(output_path, f"torch_back_{theta}.jpg"), torch_back_image
        )

        cv2.imwrite(
            os.path.join(output_path, f"torch_diff_{theta}.jpg"),
            np.abs(image - torch_back_image),
        )
        cv2.imwrite(
            os.path.join(output_path, f"torch_numpy_diff_{theta}.jpg"),
            np.abs(torch_rotated_image - rotated_image),
        )
        cv2.imwrite(
            os.path.join(output_path, f"torch_numpy_back_diff_{theta}.jpg"),
            np.abs(torch_back_image - back_image),
        )


if __name__ == "__main__":
    main()
