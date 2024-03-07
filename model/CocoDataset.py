import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CocoDetection

from pycocotools.coco import COCO




class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, transform = None, target_transform = None, transforms = None):
        super(CocoDataset, self).__init__(root, annFile, transform, target_transform, transforms)

        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        image, target = super(CocoDataset, self).__getitem__(index)
        image = transforms.ToTensor()(image)

        #* Add Masks to the target
        nums = len(target)
        for i in range(nums):
            target[i]["mask"] = torch.as_tensor(self.coco.annToMask(target[i]), dtype=torch.float32)
        return image, target
