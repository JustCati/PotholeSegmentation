import os
import json
import random
import argparse

import torch
from torch.utils import data
from torchvision.transforms import v2 as T

from utils.coco import generateJSON
from model.model import getModel, trainModel

from utils.transform import GaussianNoise
from utils.coco import CocoDataset
from utils.graphs import plotSample, plotDemo, plotPerf





def generateCoco(path, args, split="train"):
    splitPath = os.path.join(path, split)
    if not os.path.exists(splitPath):
        raise ValueError(f"Path {splitPath} does not exist")

    cocoPath = os.path.join(splitPath, "cocoLabels.json")
    if not os.path.exists(cocoPath):
        args.__dict__["split"] = split
        generateJSON(args)
    return splitPath, cocoPath


def getDevice():
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device



def main():
    parser = argparse.ArgumentParser(description="Pothole Segmentation")
    parser.add_argument("--path", type=str, default=os.path.join(os.getcwd(), "data"), help="Path to the data directory")
    parser.add_argument("--target", type=str, default="images", help="Target directory (images, videos)", choices=["images", "videos"])
    parser.add_argument("--plot", action="store_true", default=False, help="Plot a sample image from the dataset with ground truth masks")
    parser.add_argument("--output", type=str, default=os.path.join(os.getcwd(), "OUTPUT"), help="Output directory for model saving")
    parser.add_argument("--demo", action="store_true", default=False, help="Run a demo of inference on a random image from the validation set")
    parser.add_argument("--perf", action="store_true", default=False, help="Plot the performance of the model")
    args = parser.parse_args()


    path = args.path
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    path = os.path.join(path, args.target)
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    modelOutputPath = args.output
    if not os.path.exists(modelOutputPath):
        os.makedirs(modelOutputPath)

    trainPath, trainCocoPath = generateCoco(path, args, "train")
    valPath, valCocoPath = generateCoco(path, args, "val")
    trainPath = os.path.join(trainPath, "images")
    valPath = os.path.join(valPath, "images")


    #* --------------- Create Dataset -----------------

    transform = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.GaussianBlur((5, 9), (0.1, 5)),
        GaussianNoise(p = 0.5, noise_p = 0.5, mean = 0, sigma = 50),
    ])
    train, val = CocoDataset(trainPath, trainCocoPath, transforms = transform), CocoDataset(valPath, valCocoPath)

    BATCH_SIZE = 3
    trainDataloader = data.DataLoader(train, batch_size = BATCH_SIZE, num_workers = 8, pin_memory = True, shuffle = True, collate_fn = lambda x: tuple(zip(*x)))
    valDataloader = data.DataLoader(val, batch_size = BATCH_SIZE, num_workers = 8, pin_memory = True, shuffle = True, collate_fn = lambda x: tuple(zip(*x)))

    if args.plot:
        plotSample(train)
        choice = input("Continue? [y/N]: ")
        if choice.lower() == "n" or choice == "":
            return

    #* ----------------------------------------------------


    #* --------------- Train Model -----------------

    EPOCHS = 20
    device = getDevice()
    trainLosses, valAccuracy = None, None
    model = getModel(pretrained = True, device = device).to(device)

    if not os.path.exists(os.path.join(modelOutputPath, "model.pth")):
        print("\nTraining model")

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.001)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

        model, trainLosses, valAccuracy = trainModel(model,
                                                   trainDataloader, 
                                                   valDataloader, 
                                                   optimizer, 
                                                   lr_scheduler,
                                                   EPOCHS,
                                                   path = modelOutputPath,
                                                   device = device)

        averageLoss = sum([v["total_loss"] for _, v in trainLosses.items()]) / len(trainLosses)
        averageMap = sum([v["map"] for _, v in valAccuracy.items()]) / len(valAccuracy)
        print("Average Train Loss: ", averageLoss)
        print("Average Validation mAP: ", averageMap)

    else:
        print("\nLoading model")

        model.load_state_dict(torch.load(os.path.join(modelOutputPath, "model.pth")))
        if os.path.exists(os.path.join(modelOutputPath, "TrainLosses.json")):
            with open(os.path.join(modelOutputPath, "TrainLosses.json"), "r") as f:
                trainLosses = json.load(f)
        if os.path.exists(os.path.join(modelOutputPath, "ValAccuracy.json")):
            with open(os.path.join(modelOutputPath, "ValAccuracy.json"), "r") as f:
                valAccuracy = json.load(f)

    #* ----------------------------------------------------


    #* --------------- Plot inferenced example -----------------

    BBOX_THRESHOLD = 0.7
    MASK_THRESHOLD = 0.7

    if args.demo:
        (img, target) = val[random.randint(0, len(val) - 1)]

        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])
            prediction = {k: v.to("cpu") for k, v in prediction[0].items()}

        #! Thresholding for visualization (PlotDemo shows only the pixels with 1)
        for i in range(len(prediction['boxes'])):
            prediction['scores'][i] = prediction['scores'][i] > BBOX_THRESHOLD
        for i in range(len(prediction["masks"])):
            prediction["masks"][i] = prediction["masks"][i] > MASK_THRESHOLD

        plotDemo(img, target, prediction)

    #* ----------------------------------------------------


    #* --------------- Plot losses -----------------

    if args.perf:
        if trainLosses is None and valAccuracy is None:
            if not os.path.exists(os.path.join(modelOutputPath, "losses.json")) or not os.path.exists(os.path.join(modelOutputPath, "valLosses.json")):
                raise ValueError("Losses file not found")
            with open(os.path.join(modelOutputPath, "TrainLosses.json"), "r") as f:
                trainLosses = json.load(f)
            with open(os.path.join(modelOutputPath, "ValAccuracy.json"), "r") as f:
                valAccuracy = json.load(f)

        losses = dict(sorted(trainLosses.items(), key=lambda x: int(x[0])))
        accuracy = dict(sorted(valAccuracy.items(), key=lambda x: int(x[0])))
        plotPerf(losses, accuracy)

    #* ----------------------------------------------------



if __name__ == "__main__":
    main()
