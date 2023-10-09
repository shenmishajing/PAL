import os
import shutil

import cv2


def main():
    data_path = "temp/coco_dna/"
    output_path = "temp/coco_dna_output/"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    file_list = [
        [
            "other-Annotated20181206XIMEACervix-10_13_",
            [
                [0, 2577, 637, 3249],
            ],
        ],
        [
            "other-Annotated20181206XIMEACervix-10_17_",
            [
                [0, 1825, 858, 2664],
                [1100, 835, 1888, 1586],
            ],
        ],
        [
            "other-Annotated20181206XIMEACervix-10_22_",
            [
                [0, 0, 721, 701],
                [2253, 2628, 2808, 3527],
            ],
        ],
    ]

    for name, bboxes in file_list:
        for file in os.listdir(data_path):
            if file.startswith(name):
                img = cv2.imread(os.path.join(data_path, file))

                if len(bboxes) > 1:
                    for i, bbox in enumerate(bboxes):
                        cropped = img[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
                        cur_file = name + f"{i}_" + file[len(name) :]
                        cv2.imwrite(os.path.join(output_path, cur_file), cropped)
                else:
                    bbox = bboxes[0]
                    cropped = img[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
                    cv2.imwrite(os.path.join(output_path, file), cropped)


if __name__ == "__main__":
    main()
