# Copyright (c) OpenMMLab. All rights reserved.
import multiprocessing
import timeit
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Sequence

import numpy as np
import pycocotools.mask as mask_util
import torch
from mmdet.evaluation import CocoMetric as _CocoMetric
from mmdet.registry import METRICS


def do_encode_mask_results(mask):
    return mask_util.encode(np.array(mask[:, :, np.newaxis], order="F", dtype="uint8"))[
        0
    ]


@METRICS.register_module(force=True)
class CocoMetric(_CocoMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encode_mask_results_pool = None

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        if self.encode_mask_results_pool is None:
            self.encode_mask_results_pool = multiprocessing.Pool()

        cur_result = []
        for data_sample in data_samples:
            result = dict()
            pred = data_sample["pred_instances"]
            result["img_id"] = data_sample["img_id"]
            result["bboxes"] = pred["bboxes"].cpu().numpy()
            result["scores"] = pred["scores"].cpu().numpy()
            result["labels"] = pred["labels"].cpu().numpy()
            # encode mask to RLE
            if "masks" in pred:
                result["masks"] = (
                    self.encode_mask_results_pool.map_async(
                        do_encode_mask_results, pred["masks"].detach().cpu().numpy()
                    )
                    if isinstance(pred["masks"], torch.Tensor)
                    else pred["masks"]
                )
            # some detectors use different scores for bbox and mask
            if "mask_scores" in pred:
                result["mask_scores"] = pred["mask_scores"].cpu().numpy()

            # parse gt
            gt = dict()
            gt["width"] = data_sample["ori_shape"][1]
            gt["height"] = data_sample["ori_shape"][0]
            gt["img_id"] = data_sample["img_id"]
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert "instances" in data_sample, (
                    "ground truth is required for evaluation when "
                    "`ann_file` is not provided"
                )
                gt["anns"] = data_sample["instances"]
            # add converted result to the results list
            cur_result.append((gt, result))

        for r in cur_result:
            if "masks" in r[1] and isinstance(
                r[1]["masks"], multiprocessing.pool.AsyncResult
            ):
                r[1]["masks"] = r[1]["masks"].get()

        self.results.extend(cur_result)

    def evaluate(self, *args, **kwargs) -> dict:
        if self.encode_mask_results_pool is not None:
            self.encode_mask_results_pool.close()
            self.encode_mask_results_pool.join()
            self.encode_mask_results_pool = None
        return super().evaluate(*args, **kwargs)


def encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code.

    Args:
        mask_results (list): bitmap mask results.

    Returns:
        list | tuple: RLE encoded mask.
    """
    encoded_mask_results = []
    for mask in mask_results:
        encoded_mask_results.append(
            mask_util.encode(
                np.array(mask[:, :, np.newaxis], order="F", dtype="uint8")
            )[0]
        )  # encoded with RLE
    return encoded_mask_results


def pycoco_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code.

    Args:
        mask_results (list): bitmap mask results.

    Returns:
        list | tuple: RLE encoded mask.
    """
    return mask_util.encode(
        mask_results.transpose(1, 2, 0).astype(np.uint8, "F", copy=False)
    )


def expand_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code.

    Args:
        mask_results (list): bitmap mask results.

    Returns:
        list | tuple: RLE encoded mask.
    """
    encoded_mask_results = []
    for mask in mask_results:
        encoded_mask_results.append(
            mask_util.encode(
                np.array(np.expand_dims(mask, axis=-1), order="F", dtype="uint8")
            )[0]
        )  # encoded with RLE
    return encoded_mask_results


def do_encode_mask_results(mask):
    return mask_util.encode(np.array(mask[:, :, np.newaxis], order="F", dtype="uint8"))[
        0
    ]


def multi_process_encode_mask_results(mask_results, pool):
    """Encode bitmap mask to RLE code.

    Args:
        mask_results (list): bitmap mask results.

    Returns:
        list | tuple: RLE encoded mask.
    """
    return pool.map(do_encode_mask_results, mask_results)


def executor_encode_mask_results(mask_results, excuter):
    """Encode bitmap mask to RLE code.

    Args:
        mask_results (list): bitmap mask results.

    Returns:
        list | tuple: RLE encoded mask.
    """
    return list(excuter.map(do_encode_mask_results, mask_results))


def time_func(func_name, arg_str="mask_results", **kwargs):
    times = timeit.repeat(
        f"{func_name}({arg_str})", f"from __main__ import {func_name}", **kwargs
    )
    if isinstance(times, list) and len(times) > 1:
        m = np.mean(times)
        s = np.std(times)
    else:
        m = times
        s = 0
    print(f"{func_name}, mean={m}, std={s}")


def speed_benchmark():
    kwargs = dict(
        globals=dict(
            mask_results=np.random.random((100, 4000, 2000)) > 0.5,
            pool=multiprocessing.Pool(),
            thread_pool=ThreadPoolExecutor(48),
            process_pool=ProcessPoolExecutor(),
        ),
        repeat=5,
        number=1,
    )

    time_func("encode_mask_results", **kwargs)
    time_func("pycoco_encode_mask_results", **kwargs)
    time_func("expand_encode_mask_results", **kwargs)
    time_func(
        "multi_process_encode_mask_results", arg_str="mask_results, pool", **kwargs
    )
    time_func(
        "executor_encode_mask_results", arg_str="mask_results, thread_pool", **kwargs
    )
    time_func(
        "executor_encode_mask_results", arg_str="mask_results, process_pool", **kwargs
    )


def correct_assert():
    mask_results = np.random.random((100, 4000, 2000)) > 0.5

    res = encode_mask_results(mask_results)
    assert res == pycoco_encode_mask_results(mask_results)
    assert res == expand_encode_mask_results(mask_results)
    assert res == multi_process_encode_mask_results(
        mask_results, multiprocessing.Pool()
    )
    assert res == executor_encode_mask_results(mask_results, ThreadPoolExecutor(48))
    assert res == executor_encode_mask_results(mask_results, ProcessPoolExecutor())


def main():
    speed_benchmark()
    correct_assert()


if __name__ == "__main__":
    main()
