import os
import json
import argparse

import torch
from torch.utils import data
from torchvision.transforms import v2 as T
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP

from utils.coco import generateJSON
from model.train import trainModel
from model.evaluate import evaluate_one_epoch
from model.model import getModel, loadCheckpoint

from utils.coco import CocoDataset
from utils.graphs import plotSample, plotDemo, plotPerf
from utils.transform import GaussianNoise, GaussianBlur





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
    parser.add_argument("--sample", action="store_true", default=False, help="Plot a sample image from the dataset with ground truth masks")
    parser.add_argument("--output", type=str, default=os.path.join(os.getcwd(), "OUTPUT"), help="Output directory for model saving")
    parser.add_argument("--demo", action="store_true", default=False, help="Run a demo of inference on a random image from the validation set")
    parser.add_argument("--perf", action="store_true", default=False, help="Plot the performance of the model")
    parser.add_argument("--train", action="store_true", default=False, help="Force Training of the model")
    parser.add_argument("--eval", action="store_true", default=False, help="Evaluate the model")
    args = parser.parse_args()


    path = args.path
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    path = os.path.join(path, args.target)
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    modelOutputPath = os.path.join(args.output, "mask_rcnn_" + datetime.datetime.fromtimestamp(int(time.time())))
    if not os.path.exists(modelOutputPath):
        os.makedirs(modelOutputPath)

    trainPath, trainCocoPath = generateCoco(path, args, "train")
    valPath, valCocoPath = generateCoco(path, args, "val")
    trainPath = os.path.join(trainPath, "images")
    valPath = os.path.join(valPath, "images")


    #* --------------- Create Dataset -----------------

    #! Uncomment Gaussian Noise but performance will suffer a lot
    transform = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        GaussianBlur(0.5, (5, 9), (0.1, 5)),
        # GaussianNoise(p = 0.5, noise_p = 0.07, mean = 0, sigma = 5),
    ])

    val = CocoDataset(valPath, valCocoPath)
    train = CocoDataset(trainPath, trainCocoPath, transforms = transform)

    BATCH_SIZE = 3
    trainDataloader = data.DataLoader(train, 
                                      batch_size = BATCH_SIZE, 
                                      num_workers = 8, 
                                      pin_memory = True, 
                                      shuffle = True, 
                                      collate_fn = lambda x: tuple(zip(*x)))
    valDataloader = data.DataLoader(val, 
                                    batch_size = 1, 
                                    num_workers = 8, 
                                    pin_memory = True, 
                                    shuffle = False, 
                                    collate_fn = lambda x: tuple(zip(*x)))

    if args.sample:
        plotSample(train)
        choice = input("Continue? [y/N]: ")
        if choice.lower() == "n" or choice == "":
            return

    #* ----------------------------------------------------


    #* --------------- Train Model -----------------

    EPOCHS = 50
    BBOX_THRESHOLD = 0.7
    MASK_THRESHOLD = 0.7

    last_epoch = 0
    device = getDevice()
    model = getModel(pretrained = True, device = device)

    if not os.path.exists(os.path.join(modelOutputPath, "model.pth")) or args.train:
        print("\nTraining model")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.001)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

        if os.path.exists(os.path.join(modelOutputPath, "model.pth")):
            print("Model found, continuing training...")
            model, optimizer, lr_scheduler, last_epoch = loadCheckpoint(model, optimizer, lr_scheduler, path = modelOutputPath, device = device)

        cfg = {
            "model" : model,
            "optimizer" : optimizer,
            "lr_scheduler" : lr_scheduler,
            "last_epoch" : last_epoch,
            "epoch" : EPOCHS,
            "mask_threshold" : MASK_THRESHOLD,
            "device" : device,
            "trainDataloader" : trainDataloader,
            "valDataloader" : valDataloader,
            "tb_writer" : tb_writer,
            "path" : modelOutputPath,
        }
        model = trainModel(cfg)

    elif os.path.exists(os.path.join(modelOutputPath, "model.pth")):
        print("\nLoading model")
        model, *_ = loadCheckpoint(model, path = modelOutputPath, device = device)
    else:
        raise ValueError("Model file not found")

    #* ----------------------------------------------------

    #* --------------- Evaluate Model -----------------
    
    if args.eval:
        print("\nEvaluating model")
        model.eval()
        with torch.no_grad():
            valAccuracy = evaluate_one_epoch(model, valDataloader, MASK_THRESHOLD, device)
            print("Validation Accuracy:")
            print(f"Segmentation mAP: {valAccuracy['segm_map']:.2f}, Bounding Box mAP: {valAccuracy['bbox_map']:.2f}")

    #* ----------------------------------------------------


    #* --------------- Plot inferenced example -----------------

    if args.demo:
        for _ in range(3):
            (img, target) = val[torch.randint(0, len(val), (1,))]

            model.eval()
            with torch.no_grad():
                prediction = model([img.to(device)])
                prediction = {k: v.to("cpu") for k, v in prediction[0].items()}

            #! Thresholding for visualization (PlotDemo shows only the pixels with 1)
            for i in range(len(prediction['boxes'])):
                prediction['scores'][i] = prediction['scores'][i] > BBOX_THRESHOLD
            for i in range(len(prediction["masks"])):
                prediction["masks"][i] = prediction["masks"][i] > MASK_THRESHOLD

            pred = prediction.copy()
            pred["masks"] = pred["masks"].type(torch.uint8).reshape(-1, img.shape[-1], img.shape[-1])
            target["masks"] = target["masks"].type(torch.uint8).reshape(-1, img.shape[-1], img.shape[-1])

            map_segm = MAP(box_format="xyxy", iou_type="segm")
            map_segm.update([pred], [target])
            segm_acc = map_segm.compute()

            map_bbox = MAP(box_format="xyxy", iou_type="bbox")
            map_bbox.update([pred], [target])
            bbox_acc = map_bbox.compute()

            print("----------------------------------------------")
            print("DEMO Segmentation mAP:")
            print(f"Mean Average Precision: {segm_acc['map']:.2f}, Mean Average Precision (50): {segm_acc['map_50']:.2f}")
            print("Bounding Box mAP:")
            print(f"Mean Average Precision: {bbox_acc['map']:.2f}, Mean Average Precision (50): {bbox_acc['map_50']:.2f}")
            plotDemo(img, target, prediction)

    #* ----------------------------------------------------


    #* --------------- Plot losses -----------------

    if args.perf:
        trainLosses, valAccuracy = None, None
        if not os.path.exists(os.path.join(modelOutputPath, "TrainLosses.json")) or \
            not os.path.exists(os.path.join(modelOutputPath, "ValAccuracy.json")):
            raise ValueError("Performance files not found")

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
