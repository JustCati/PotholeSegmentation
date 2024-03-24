import numpy as np
from torchvision import transforms

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import torch
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes



POTHOLE_CLASS = 1

def plotSample(dataset):
    (img, target) = dataset[torch.randint(0, len(dataset), (1,))]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.zeros((640, 640)), interpolation='none', aspect='auto')
    for i in range(len(target["masks"])):
        alpha = 0.5 * (target["masks"][i] > 0)
        plt.imshow(target["masks"][i], alpha=alpha, interpolation='none', aspect='auto')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(transforms.ToPILImage()(img))
    for i in range(len(target['boxes'])):
        box = target['boxes'][i]
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='w', facecolor='none'))

    for i in range(len(target["masks"])):
        alpha = 0.5 * (target["masks"][i] > 0)
        plt.imshow(target["masks"][i], alpha=alpha, interpolation='none', aspect='auto')
    plt.tight_layout()
    plt.show()



def plotDemo(img, target, prediction):
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(transforms.ToPILImage()(img), aspect='auto')

    plt.subplot(1, 3, 2)
    image = (img * 255).type(torch.uint8)
    targetMasks = target["masks"].type(torch.bool).reshape(-1, img.shape[-1], img.shape[-1])
    image = draw_segmentation_masks(image, targetMasks, alpha=0.5, colors="yellow")
    image = draw_bounding_boxes(image, target["boxes"], colors="white", width=3)
    plt.imshow(transforms.ToPILImage()(image), aspect='auto')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    img = (img * 255).type(torch.uint8)
    masks = prediction["masks"].type(torch.bool).reshape(-1, img.shape[-1], img.shape[-1])
    img = draw_bounding_boxes(img, prediction["boxes"], colors="red", width=3)
    img = draw_bounding_boxes(img, target["boxes"], colors="white", width=3)
    img = draw_segmentation_masks(img.type(torch.uint8), masks, alpha=0.5, colors="red")
    plt.imshow(transforms.ToPILImage()(img), aspect='auto')
    plt.axis('off')

    plt.tight_layout()
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
    plt.plot([acc["segm_map"] for _, acc in accuracy], label="Segmentation mAP")
    plt.plot([acc["bbox_map"] for _, acc in accuracy], label="Box mAP")
    plt.scatter([i for i, _ in accuracy], [acc["segm_map"] for _, acc in accuracy], color="red", marker="s", s=10)
    plt.scatter([i for i, _ in accuracy], [acc["bbox_map"] for _, acc in accuracy], color="red",marker="s", s=10)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.grid(True)
    plt.xticks(range(0, len(accuracy), 1))
    plt.legend()

    plt.show()
