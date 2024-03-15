import os
import json
import math
import cv2 as cv
import pandas as pd

import numpy as np
from PIL import Image

import torch
from torchvision import transforms, tv_tensors
from torchvision.datasets import VisionDataset

from pycocotools.coco import COCO






class CocoDataset(VisionDataset):
    def __init__(self, path, annFile, transform = None, target_transform = None, transforms = None):
        super().__init__(path, transforms, transform, target_transform)
        
        self.root = path
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)


        #* Load the image
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = transforms.ToTensor()(img)

        #* Load and convert to tensor the annotations
        nums = len(target)
        boxes = []
        for i in range(nums):
            xmin, ymin, width, height = target[i]['bbox']
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])

        areas = [target[i]['area'] for i in range(nums)]
        masks = np.array([coco.annToMask(target[i]) for i in range(nums)])

        img_id = torch.tensor([img_id])
        labels = torch.ones((nums,), dtype=torch.int64)
        is_crowd = torch.zeros((nums,), dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        boxes = tv_tensors.BoundingBoxes(boxes, format='XYXY', canvas_size=img.shape[-2:])
        masks = tv_tensors.Mask(masks)

        target = {
            "image_id": img_id,
            "labels": labels,
            "area": areas,
            "iscrowd": is_crowd,
            "boxes": boxes,
            "masks": masks
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def __len__(self):
        return len(self.ids)





def seg_to_bbox(seg_info):
    _, *points = seg_info.strip().split()
    points = [float(p) for p in points]
    x_min, y_min, x_max, y_max = min(points[0::2]), min(points[1::2]), max(points[0::2]), max(points[1::2])
    return [x_min, y_min, x_max - x_min, y_max - y_min]



def generateJSON(args):
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

    cocoTrainPath = os.path.join(path, "cocoLabels.json")
    coco = {"images": [], "annotations": [], "categories": [{"supercategory": "pothole", "id": 1, "name": "pothole"}]}

    imgID = 0
    annID = 0
    for _, row in trainDF.iterrows():
        image = cv.imread(os.path.join(imagePath, row["imageFile"] + ".jpg"))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        xlim = math.ceil(image.shape[1])
        ylim = math.ceil(image.shape[0])

        coco["images"].append({
            "id": imgID,
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
                    "id": annID,
                    "image_id": imgID,
                    "bbox": bbox,
                    "bbox_mode": "XYWH",
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3],
                    "category_id": coco["categories"][0]["id"],
                    "segmentation": [seg],
                })
                annID += 1
        imgID += 1
    image = None

    with open(cocoTrainPath, "w") as f:
        json.dump(coco, f)
