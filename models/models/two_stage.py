from mmdet.models import TwoStageDetector as _TwoStageDetector
from mmengine.config import ConfigDict


class TwoStageDetector(_TwoStageDetector):
    def __init__(self, *args, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, dict):
                kwargs[k] = ConfigDict(v)
        super().__init__(**kwargs)
