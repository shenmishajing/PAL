from mmdet.structures import SampleList
from torch import Tensor

from .two_stage import TwoStageDetector
from .utils.predict_ellipse_segmentation import predict_ellipse_segmentation


class TwoStagePredictEllipseSegmentationDetector(TwoStageDetector):
    def predict(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    ) -> SampleList:
        batch_data_samples = super().predict(batch_inputs, batch_data_samples, rescale)
        predict_ellipse_segmentation(batch_data_samples)
        return batch_data_samples
