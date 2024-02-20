
import os
import json
import math
import argparse
import cv2 as cv
import pandas as pd
from detectron2.structures import BoxMode


parser = argparse.ArgumentParser(description='Script for conversion of YoloV8 labels to COCO format.')
parser.add_argument("--path", type=str, default=os.path.join(os.getcwd(), "data"), help="Path to the data directory")
parser.add_argument("--split", type=str, default="train", help="Target directory (train, test, val)", choices=["train", "test", "val"])
parser.add_argument("--target", type=str, default="images", help="Target directory (images, videos)", choices=["images", "videos"])
args = parser.parse_args()



def convert(args):
    path = args.path
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    
    path = os.path.join(path, args.target, args.split)
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    imagePath = os.path.join(path, "images")
    labelsPath = os.path.join(path, "labels")

    
    trainImage = sorted(os.listdir(imagePath))
    trainLabels = sorted(os.listdir(labelsPath))

    if len(trainImage) != len(trainLabels):
        raise ValueError(f"Number of images and labels do not match")

    for i in range(len(trainImage)):
        trainImage[i] = trainImage[i][:-4]
        trainLabels[i] = trainLabels[i][:-4]

    imageDF = pd.DataFrame({"imageFile": trainImage})
    labelDF = pd.DataFrame({"labelFile": trainLabels})

    trainDF = pd.merge(imageDF, labelDF, left_on="imageFile", right_on="labelFile", how="inner")
    trainDF = trainDF.sample(frac=1).reset_index(drop=True)

    
    def seg_to_bbox(seg_info):
        _, *points = seg_info.strip().split()
        points = [float(p) for p in points]
        x_min, y_min, x_max, y_max = min(points[0::2]), min(points[1::2]), max(points[0::2]), max(points[1::2])
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    
    
    cocoTrainPath = os.path.join(path, "cocoLabels.json")
    coco = {"images": [], "annotations": [], "categories": [{"supercategory": "pothole", "id": 1, "name": "pothole"}]}


    for _, row in trainDF.iterrows():
        image = cv.imread(os.path.join(imagePath, row["imageFile"] + ".jpg"))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        xlim = math.ceil(image.shape[1])
        ylim = math.ceil(image.shape[0])

        coco["images"].append({
            "id": row["imageFile"],
            "width": xlim,
            "height": ylim,
            "file_name": row["imageFile"] + ".jpg"
        })

        with open(os.path.join(labelsPath, row["labelFile"] + ".txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                bbox = seg_to_bbox(line)
                bbox = [p * xlim if i % 2 == 0 else p * ylim for i, p in enumerate(bbox)]

                seg = [float(p) for p in line.strip().split(" ")[1:]]
                seg = [p * xlim if i % 2 == 0 else p * ylim for i, p in enumerate(seg)]

                coco["annotations"].append({
                    "image_id": row["imageFile"],
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": coco["categories"][0]["id"],
                    "segmentation": seg,
                })
    image = None
    
    with open(cocoTrainPath, "w") as f:
        json.dump(coco, f)




if __name__ == "__main__":
    convert(args)
