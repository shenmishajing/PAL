from lightning.pytorch.cli import instantiate_class
from mmlab_lightning.datasets import MMLabDataSetAdapter


class MMDetDataSetAdapter(MMLabDataSetAdapter):
    def _build_batch_sampler(self, batch_sampler_cfg, dataset, *args):
        return instantiate_class((dataset,) + args, batch_sampler_cfg)
