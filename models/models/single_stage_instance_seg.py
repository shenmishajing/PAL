from mmdet.models.detectors.single_stage_instance_seg import \
    SingleStageInstanceSegmentor as _SingleStageInstanceSegmentor
from mmengine.config import ConfigDict


class SingleStageInstanceSegmentor(_SingleStageInstanceSegmentor):
    def __init__(self, *args, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, dict):
                kwargs[k] = ConfigDict(v)
        super().__init__(**kwargs)
