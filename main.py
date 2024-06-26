import os
import time
import random
import datetime
import argparse

import numpy as np
import pandas as pd

import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
from torchvision.transforms import v2 as T

from torch.utils.tensorboard import SummaryWriter

from libs.model.train import trainModel
from libs.model.model import getModel, loadCheckpoint
from libs.model.evaluate import evaluate_one_epoch, demo
from libs.data.coco import generateJSON, worker_reset_seed


from libs.data.coco import CocoDataset
from libs.graphs import plotSample, plotDemo, plotPerf
from libs.data.transform import GaussianNoise, GaussianBlur





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


def fix_random_seed(seed):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    #! Commented because roi__align has a deterministic versione very memory hungry 
    # torch.use_deterministic_algorithms(True) #! https://github.com/pytorch/vision/issues/8168#issuecomment-1890599205
    return rng_generator



def main(args):
    path = args.path
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    path = os.path.join(path, "images")
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    modelOutputPath = ""
    if args.train:
        modelOutputPath = os.path.join(os.getcwd(), "OUTPUT", "mask_rcnn_" + str(datetime.datetime.fromtimestamp(int(time.time()))))
        if not os.path.exists(modelOutputPath):
            os.makedirs(modelOutputPath)
    elif args.resume:
        modelOutputPath = args.resume
    elif args.perf or args.eval or args.demo:
        if args.perf:
            modelOutputPath = args.perf
        elif args.eval:
            modelOutputPath = args.eval
        elif args.demo:
            modelOutputPath = args.demo

    if not os.path.exists(modelOutputPath) and not args.sample:
        raise ValueError(f"Path {modelOutputPath} does not exist")

    trainPath, trainCocoPath = generateCoco(path, args, "train")
    valPath, valCocoPath = generateCoco(path, args, "val")
    trainPath = os.path.join(trainPath, "images")
    valPath = os.path.join(valPath, "images")


    #* --------------- Create Dataset -----------------

    if args.train or args.sample or args.demo or args.eval or args.resume != "":
        SEED = 1234567891
        rng_generator = fix_random_seed(SEED)

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
                                        generator=rng_generator,
                                        worker_init_fn=worker_reset_seed,
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


    #* --------------- Train or Load Model -----------------
    EPOCHS = args.epochs if args.epochs else 10
    BBOX_THRESHOLD = 0.7
    MASK_THRESHOLD = 0.7

    curr_epoch = 0
    device = getDevice()
    model = getModel(pretrained = True, device = device)
    tb_writer = SummaryWriter(os.path.join(modelOutputPath, "logs"))

    if args.train or args.resume != "":
        print("\nTraining model")

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.001)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

        if args.resume != "" and os.path.exists(os.path.join(modelOutputPath, "checkpoint.pth")):
            print("Most recent trained model found, continuing training...")

            model, *_ = loadCheckpoint(model, path = modelOutputPath, device = device, best = True)
            print("Evaluating best model to get mAP...")
            best_Acc = evaluate_one_epoch(model, valDataloader, MASK_THRESHOLD, device, None)["segm_map"]

            #* Load last checkpoint
            print("Loading last checkpoint...")
            model, optimizer, lr_scheduler, curr_epoch = loadCheckpoint(model, optimizer, lr_scheduler, path = modelOutputPath, device = device, best = False)

        cfg = {
            "model" : model,
            "optimizer" : optimizer,
            "lr_scheduler" : lr_scheduler,
            "curr_epoch" : curr_epoch,
            "epoch" : curr_epoch + (EPOCHS - curr_epoch),
            "mask_threshold" : MASK_THRESHOLD,
            "device" : device,
            "trainDataloader" : trainDataloader,
            "valDataloader" : valDataloader,
            "tb_writer" : tb_writer,
            "path" : modelOutputPath,
            "best_Acc" : best_Acc if args.resume != "" else float("-inf")
        }
        trainModel(cfg)

    elif os.path.exists(os.path.join(modelOutputPath, "model.pth")) and args.perf == "":
        print("\nLoading best model")
        model, _, _, epoch = loadCheckpoint(model, path = modelOutputPath, device = device, best = True)
        print(f"Loaded best model at epoch {epoch + 1}")
    elif args.perf == "":
            raise ValueError("Model file not found")
    #* ----------------------------------------------------

    #* --------------- Evaluate Model -----------------
    if args.eval:
        print("\nEvaluating model")
        evaluate_one_epoch(model, valDataloader, MASK_THRESHOLD, device, None)
    #* ----------------------------------------------------


    #* --------------- Plot inferenced example -----------------
    if args.demo:
        if not args.save:
            for _ in range(3):
                (img, target) = val[torch.randint(0, len(val), (1,))]

                cfg = {
                    "model" : model,
                    "img" : img,
                    "target" : target,
                    "MASK_THRESHOLD" : MASK_THRESHOLD,
                    "BBOX_THRESHOLD" : BBOX_THRESHOLD,
                    "device" : device
                }
                pred = demo(**cfg)
                plotDemo(**pred)
        else:
            demoPath = os.path.join(modelOutputPath, "demo")
            if not os.path.exists(demoPath):
                os.makedirs(demoPath)
            for idx, (img, target) in enumerate(val):
                cfg = {
                    "model" : model,
                    "img" : img,
                    "target" : target,
                    "MASK_THRESHOLD" : MASK_THRESHOLD,
                    "BBOX_THRESHOLD" : BBOX_THRESHOLD,
                    "device" : device
                }
                pred = demo(**cfg)
                outPath = os.path.join(demoPath, f"demo_{idx}.png")
                plotDemo(**pred, save=True, path=outPath)
                del pred
                del cfg
    #* ----------------------------------------------------


    #* --------------- Plot losses -----------------
    if args.perf:
        if not os.path.exists(os.path.join(modelOutputPath, "csv")):
            raise ValueError("Performance files not found")

        bbox_loss = pd.read_csv(os.path.join(modelOutputPath, "csv", "logs_train_all_losses_loss_box_reg.csv")).drop(columns = ["Wall time", "Step"])
        segm_loss = pd.read_csv(os.path.join(modelOutputPath, "csv", "logs_train_all_losses_loss_mask.csv")).drop(columns = ["Wall time", "Step"])
        final_loss = pd.read_csv(os.path.join(modelOutputPath, "csv", "logs.csv")).drop(columns = ["Wall time", "Step"])

        bbox_map = pd.read_csv(os.path.join(modelOutputPath, "csv", "logs_val_map_bbox_map.csv")).drop(columns = ["Wall time", "Step"])
        segm_map = pd.read_csv(os.path.join(modelOutputPath, "csv", "logs_val_map_segm_map.csv")).drop(columns = ["Wall time", "Step"])

        args = {
            "bbox_loss" : bbox_loss,
            "segm_loss" : segm_loss,
            "final_loss" : final_loss,
            "bbox_map" : bbox_map,
            "segm_map" : segm_map
        }
        plotPerf(args)
    #* ----------------------------------------------------



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pothole Segmentation")
    parser.add_argument("--path", type=str, default=os.path.join(os.getcwd(), "data", "Potholes"), help="Path to the data directory")
    parser.add_argument("--sample", action="store_true", default=False, help="Plot a sample image from the dataset with ground truth masks")
    parser.add_argument("--train", action="store_true", default=False, help="Force Training of the model")
    parser.add_argument("--demo", type=str, default="", help="Run a demo of inference on 3 random image from the validation set with the model at the specified path")
    parser.add_argument("--perf", type=str, default="", help="Plot the performance of the model at the specified path")
    parser.add_argument("--eval", type=str, default="", help="Evaluate the model at the specified path")
    parser.add_argument("--save", action="store_true", default=False, help="Save the demo images to the model directory")
    parser.add_argument("--resume", type=str, default="", help="Resume training from the specified model path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    args = parser.parse_args()

    main(args)
