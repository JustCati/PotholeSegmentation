import os
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
        boxes = [target[i]['bbox'] for i in range(nums)]

        areas = [target[i]['area'] for i in range(nums)]
        masks = np.array([coco.annToMask(target[i]) for i in range(nums)])

        img_id = torch.tensor([img_id])
        labels = torch.ones((nums,), dtype=torch.int64)
        is_crowd = torch.zeros((nums,), dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        boxes = tv_tensors.BoundingBoxes(boxes, format='XYWH', canvas_size=img.shape[-2:])
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
