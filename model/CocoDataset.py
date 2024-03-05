import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms

from pycocotools.coco import COCO


class MaskDataset(data.Dataset):
    def __init__(self, path, annFile):
        self.root = path
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())

    def _combineMasks(self, data):
        ret = torch.cat(data, dim=0)
        ret = torch.sum(ret, dim=0)
        ret = (ret > 0).float()
        return ret

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        #* Load the image
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1)))
        ])
        img = transformer(img)

        #* Load Masks
        masks = []
        nums = len(target)
        elem = [coco.annToMask(target[i]) for i in range(nums)]
        for i in range(nums):
            masks.append(transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])(elem[i]))
        return img, torch.as_tensor(self._combineMasks(masks))

    def __len__(self):
        return len(self.ids)



class CocoDataset(data.Dataset):
    def __init__(self, path, annFile):
        self.root = path
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())

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
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.float32)

        target = {
            "image_id": img_id,
            "labels": labels,
            "area": areas,
            "iscrowd": is_crowd,
            "boxes": boxes,
            "masks": masks
        }
        return img, target


    def __len__(self):
        return len(self.ids)
