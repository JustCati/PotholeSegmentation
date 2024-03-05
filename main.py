import os
import json
import random
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
from torch.utils import data
from torchvision import transforms

from coco.coco import generateJSON
from model.model import getModel, trainModel
from model.CocoDataset import CocoDataset, MaskDataset




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
    parser.add_argument("--output", type=str, default=os.path.join(os.getcwd(), "OUTPUT"), help="Output directory for model saving")
    parser.add_argument("--model", type=str, default="MASK", help="Model to use (MASK, UNET)", choices=["MASK", "UNET"])
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

    train, val = None, None
    BATCH_SIZE = 3 if args.model == "MASK" else 25

    if args.model == "MASK":
        train = CocoDataset(trainPath, trainCocoPath)
        val = CocoDataset(valPath, valCocoPath)
    if args.model == "UNET":
        train = MaskDataset(trainPath, trainCocoPath)
        val = MaskDataset(valPath, valCocoPath)

    dataloader = data.DataLoader(train, batch_size = BATCH_SIZE, num_workers = 8, pin_memory = True, shuffle = True, collate_fn = lambda x: tuple(zip(*x)))

    if args.plot:
        plotSample(train)

    #* ----------------------------------------------------


    #* --------------- Train Model -----------------

    EPOCHS = 20

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = getModel(args.model, pretrained = True, device = device)
    model.to(device)
    
    if not os.path.exists(os.path.join(modelOutputPath, f"model_{args.model}.pth")):
        print("\nTraining model")

        params = [p for p in model.parameters() if p.requires_grad]
        loss_func = torch.nn.BCEWithLogitsLoss() if args.model == "UNET" else None
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        model, losses = trainModel(model, dataloader, optimizer, lr_scheduler, EPOCHS, loss_func = loss_func if args.model == "UNET" else None, path = modelOutputPath, device = device)
        print("Average Loss: ", sum([value["total_loss"] for value in losses[0] if isinstance(value, dict)]) / len(losses))
    else:
        model.load_state_dict(torch.load(os.path.join(modelOutputPath, f"model_{args.model}.pth")))
        with open(os.path.join(modelOutputPath, f"losses_{args.model}.json"), "r") as f:
            losses = json.load(f)

    #* ----------------------------------------------------


    #* --------------- Plot inferenced example -----------------

    N_EXAMPLES = 0
    for _ in range(N_EXAMPLES):
        (img, target) = val[random.randint(0, len(val) - 1)]

        model.eval()
        with torch.no_grad():
            if args.model == "MASK":
                prediction = model([img.to(device)])
                prediction = {k: v.to("cpu") for k, v in prediction[0].items()}
            else:
                prediction = model(img.unsqueeze(0).to(device))
                prediction = prediction.to("cpu")

        plt.axis('off')
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        if args.model == "MASK":
            plt.imshow(transforms.ToPILImage()(img))
        else:
            plt.imshow(img[0], interpolation='none')

        plt.subplot(1, 2, 2)
        if args.model == "MASK":
            plt.imshow(transforms.ToPILImage()(img))
            for i in range(len(target['boxes'])):
                box = target['boxes'][i]
                plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='w', facecolor='none'))

            for i in range(len(prediction['boxes'])):
                if prediction['scores'][i] > 0.7 and prediction['labels'][i] == 1: #? Filter out non-potholes
                    box = prediction['boxes'][i]
                    plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none'))

            for i in range(len(target["masks"])):
                alpha = 0.5 * (target["masks"][i] > 0)
                plt.imshow(target["masks"][i], alpha=alpha, interpolation='none')

        if args.model == "UNET":
            plt.imshow(prediction[0][0], alpha=0.5, interpolation='none')

        plt.show()

    #* ----------------------------------------------------
    
    #* --------------- Plot losses -----------------

    if losses is None:
        if not os.path.exists(os.path.join(modelOutputPath, f"losses_{args.model}.json")):
            raise ValueError("Losses file not found")
        with open(os.path.join(modelOutputPath, f"losses_{args.model}.json"), "r") as f:
            total_losses = json.load(f)

    plt.figure(figsize=(10, 5))
    total_losses = sorted(losses, key=lambda x: x[0])
    plt.plot([loss["total_loss"] for _, loss in total_losses], label="Total Loss")
    plt.plot([loss["loss_mask"] for _, loss in total_losses], label="Mask Loss")
    plt.plot([loss["loss_box_reg"] for _, loss in total_losses], label="Box Loss")
    plt.scatter([i for i, _ in total_losses], [loss["total_loss"] for _, loss in total_losses], color="red")
    plt.scatter([i for i, _ in total_losses], [loss["loss_mask"] for _, loss in total_losses], color="red")
    plt.scatter([i for i, _ in total_losses], [loss["loss_box_reg"] for _, loss in total_losses], color="red")
    
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    #* ----------------------------------------------------

if __name__ == "__main__":
    main()
