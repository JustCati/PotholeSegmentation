import random

import numpy as np
from torchvision import transforms

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


import torch

def plotSample(dataset):
    (img, target) = dataset[random.randint(0, len(dataset))]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.zeros((640, 640)), interpolation='none')
    for i in range(len(target["masks"])):
        alpha = 0.5 * (target["masks"][i] > 0)
        plt.imshow(target["masks"][i], alpha=alpha, interpolation='none')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(transforms.ToPILImage()(img))
    for i in range(len(target['boxes'])):
        box = target['boxes'][i]
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='w', facecolor='none'))

    for i in range(len(target["masks"])):
        alpha = 0.5 * (target["masks"][i] > 0)
        plt.imshow(target["masks"][i], alpha=alpha, interpolation='none')
    plt.show()



def plotDemo(img, target, prediction):
    plt.axis('off')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(transforms.ToPILImage()(img))
    for i in range(len(target['boxes'])):
        box = target['boxes'][i]
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='w', facecolor='none'))
    for i in range(len(target["masks"])):
        alpha = 0.5 * (target["masks"][i] > 0)
        plt.imshow(target["masks"][i], alpha=alpha, interpolation='none')

    plt.subplot(1, 2, 2)
    plt.imshow(transforms.ToPILImage()(img))
    for i in range(len(target['boxes'])):
        box = target['boxes'][i]
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='w', facecolor='none'))

    for i in range(len(prediction['boxes'])):
        if prediction['scores'][i] > 0.7 and prediction['labels'][i] == 1: #? Filter out non-potholes
            box = prediction['boxes'][i]
            plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none'))

    for i in range(len(target["masks"])):
        alpha = 0.9 * (target["masks"][i] > 0)
        plt.imshow(target["masks"][i], alpha=alpha, interpolation='none')

    plt.show()



def plotPerf(losses, accuracy):
    losses = losses.items()
    accuracy = accuracy.items()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot([loss["total_loss"] for _, loss in losses], label="Total Loss")
    plt.plot([loss["loss_mask"] for _, loss in losses], label="Mask Loss")
    plt.plot([loss["loss_box_reg"] for _, loss in losses], label="Box Loss")
    plt.scatter([i for i, _ in losses], [loss["total_loss"] for _, loss in losses], color="red", marker="s", s=10)
    plt.scatter([i for i, _ in losses], [loss["loss_mask"] for _, loss in losses], color="red", marker="s", s=10)
    plt.scatter([i for i, _ in losses], [loss["loss_box_reg"] for _, loss in losses], color="red", marker="s", s=10)
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.xticks(range(0, len(losses), 1))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([acc["map"] for _, acc in accuracy], label="mAP")
    plt.plot([acc["iou"] for _, acc in accuracy], label="IoU")
    plt.scatter([i for i, _ in accuracy], [acc["iou"] for _, acc in accuracy], color="red", marker="s", s=10)
    plt.scatter([i for i, _ in accuracy], [acc["map"] for _, acc in accuracy], color="red",marker="s", s=10)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.grid(True)
    plt.xticks(range(0, len(accuracy), 1))
    plt.legend()

    plt.show()
