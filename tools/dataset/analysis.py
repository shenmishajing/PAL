import json
import os
from collections import defaultdict

from numpy import mean


def average_analysis():
    data_path = "data/coco_dna/annotations"

    ann_list = ["train.json"]

    for ann_name in ann_list:
        data = json.load(open(os.path.join(data_path, ann_name)))

        res = defaultdict(int)
        for ann in data["annotations"]:
            res[ann["image_id"]] += 1

        print(f"Dataset: {ann_name}")
        print(f"Number of images: {len(data['images'])}")
        print(f"Number of annotations: {len(data['annotations'])}")
        print(f"Number of average objects per image: {mean(list(res.values()))}")
        print(f"Number of max objects per image: {max(res.values())}")
        print(f"Number of categories: {len(data['categories'])}")


def filter_livecell():
    max_ann = 300
    data_path = "data/livecell-dataset/annotations/LIVECell"
    output_path = data_path + "_filtered"
    os.makedirs(output_path, exist_ok=True)

    ann_list = [
        "livecell_coco_train.json",
        "livecell_coco_val.json",
        "livecell_coco_test.json",
    ]

    for ann_name in ann_list:
        data = json.load(open(os.path.join(data_path, ann_name)))

        ann_num = defaultdict(int)

        for ann in data["annotations"]:
            ann_num[ann["image_id"]] += 1

        data["images"] = [
            img for img in data["images"] if ann_num[img["id"]] <= max_ann
        ]

        json.dump(data, open(os.path.join(output_path, ann_name), "w"))


def main():
    average_analysis()


if __name__ == "__main__":
    main()
