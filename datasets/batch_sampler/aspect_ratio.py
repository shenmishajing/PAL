from typing import Sequence

from mmdet.datasets import AspectRatioBatchSampler as _AspectRatioBatchSampler


class AspectRatioBatchSampler(_AspectRatioBatchSampler):
    def __init__(
        self, dataset, sampler, batch_size: int, drop_last: bool = False
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)

        self.dataset = dataset

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info = self.dataset.get_data_info(idx)
            width, height = data_info["width"], data_info["height"]
            bucket_id = 0 if width < height else 1
            bucket = self._aspect_ratio_buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the bucket
        left_data = self._aspect_ratio_buckets[0] + self._aspect_ratio_buckets[1]
        self._aspect_ratio_buckets = [[] for _ in range(2)]
        while len(left_data) > 0:
            if len(left_data) <= self.batch_size:
                if not self.drop_last:
                    yield left_data[:]
                left_data = []
            else:
                yield left_data[: self.batch_size]
                left_data = left_data[self.batch_size :]
