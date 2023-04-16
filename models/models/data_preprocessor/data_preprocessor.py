import torch
import torch.nn.functional as F
from mmdet.models import BoxInstDataPreprocessor
from mmdet.registry import MODELS

try:
    import skimage
except ImportError:
    skimage = None


@MODELS.register_module()
class RotateBoxInstDataPreprocessor(BoxInstDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> dict:
        """Get pseudo mask labels using color similarity."""
        det_data = super(BoxInstDataPreprocessor, self).forward(data, training)
        inputs, data_samples = det_data["inputs"], det_data["data_samples"]

        if training:
            # get image masks and remove bottom pixels
            b_img_h, b_img_w = data_samples[0].batch_input_shape
            img_masks = []
            for i in range(inputs.shape[0]):
                img_h, img_w = data_samples[i].img_shape
                img_mask = inputs.new_ones((img_h, img_w))
                pixels_removed = int(
                    self.bottom_pixels_removed * float(img_h) / float(b_img_h)
                )
                if pixels_removed > 0:
                    img_mask[-pixels_removed:, :] = 0
                pad_w = b_img_w - img_w
                pad_h = b_img_h - img_h
                img_mask = F.pad(img_mask, (0, pad_w, 0, pad_h), "constant", 0.0)
                img_masks.append(img_mask)
            img_masks = torch.stack(img_masks, dim=0)
            start = int(self.mask_stride // 2)
            img_masks = img_masks[
                :, start :: self.mask_stride, start :: self.mask_stride
            ]

            # Get origin rgb image for color similarity
            ori_imgs = inputs * self.std + self.mean
            downsampled_imgs = F.avg_pool2d(
                ori_imgs.float(),
                kernel_size=self.mask_stride,
                stride=self.mask_stride,
                padding=0,
            )

            # Compute color similarity for pseudo mask generation
            for im_i, data_sample in enumerate(data_samples):
                # TODO: Support rgb2lab in mmengine?
                images_lab = skimage.color.rgb2lab(
                    downsampled_imgs[im_i].byte().permute(1, 2, 0).cpu().numpy()
                )
                images_lab = torch.as_tensor(
                    images_lab, device=ori_imgs.device, dtype=torch.float32
                )
                images_lab = images_lab.permute(2, 0, 1)[None]
                images_color_similarity = self.get_images_color_similarity(
                    images_lab, img_masks[im_i]
                )
                pairwise_mask = (
                    images_color_similarity >= self.pairwise_color_thresh
                ).float()

                per_im_bboxes = data_sample.gt_instances.bboxes
                if per_im_bboxes.shape[0] > 0:
                    pairwise_masks = torch.cat(
                        [pairwise_mask for _ in range(per_im_bboxes.shape[0])], dim=0
                    )
                else:
                    pairwise_masks = torch.zeros(
                        (0, self.pairwise_size**2 - 1, b_img_h, b_img_w)
                    )

                # TODO: Support BitmapMasks with tensor?
                data_sample.gt_instances.pairwise_masks = pairwise_masks
        return {"inputs": inputs, "data_samples": data_samples}
