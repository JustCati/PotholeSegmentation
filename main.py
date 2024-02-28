import os
import random
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision import transforms

from coco.coco import generateJSON
from model.CocoDataset import CocoDataset



def plotSample(dataset):
    (img, target) = dataset[random.randint(0, len(dataset))]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    for i in range(len(target["masks"])):
        plt.imshow(target["masks"][i], cmap='gray', alpha=0.5, interpolation='none')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(transforms.ToPILImage()(img))
    for i in range(len(target['boxes'])):
        box = target['boxes'][i]
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='w', facecolor='none'))
    plt.show()



def generateCoco(path, args, split="train"):
    splitPath = os.path.join(path, split)
    if not os.path.exists(splitPath):
        raise ValueError(f"Path {splitPath} does not exist")
    
    cocoPath = os.path.join(splitPath, "cocoLabels.json")
    if not os.path.exists(cocoPath):
        args.__dict__["split"] = split
        generateJSON(args)
    return splitPath, cocoPath



def main():
    parser = argparse.ArgumentParser(description="Pothole Segmentation")
    parser.add_argument("--path", type=str, default=os.path.join(os.getcwd(), "data"), help="Path to the data directory")
    parser.add_argument("--target", type=str, default="images", help="Target directory (images, videos)", choices=["images", "videos"])
    parser.add_argument("--plot", action="store_true", default=False, help="Plot a sample image from the dataset")
    args = parser.parse_args()


    path = args.path
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    path = os.path.join(path, args.target)
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    trainPath, trainCocoPath = generateCoco(path, args, "train")
    valPath, valCocoPath = generateCoco(path, args, "val")
    trainPath = os.path.join(trainPath, "images")
    valPath = os.path.join(valPath, "images")

    trainCoco = CocoDataset(trainPath, trainCocoPath)
    valCoco = CocoDataset(valPath, valCocoPath)

    if args.plot:
        plotSample(trainCoco)

    


if __name__ == "__main__":
    main()
