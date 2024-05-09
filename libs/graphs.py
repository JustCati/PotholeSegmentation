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



def plotDemo(img, target, prediction, segm_map, bbox_map, save = False, path = None):
    plt.title("mAP: {:.2f} (segm) {:.2f} (bbox)".format(segm_map, bbox_map))
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
    img = draw_bounding_boxes(img, target["boxes"], colors="white", width=3)
    img = draw_bounding_boxes(img, prediction["boxes"], colors="red", width=3)
    img = draw_segmentation_masks(img.type(torch.uint8), masks, alpha=0.5, colors="red")
    plt.imshow(transforms.ToPILImage()(img), aspect='auto')
    plt.axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()



def plotPerf(args):
    lossess = {k: v for k, v in {k: v if "loss" in k else None for k, v in args.items()}.items() if v is not None}
    maps = {k: v for k, v in {k: v if "map" in k else None for k, v in args.items()}.items() if v is not None}

    plt.figure(figsize=(10, 8))
    for idx, (key, value) in enumerate(lossess.items()):
        newvalue = value.ewm(alpha=1 - 0.99).mean()
        
        plt.subplot(len(lossess), 1, idx + 1)
        plt.plot(value, c="gray", alpha=0.5)
        plt.plot(newvalue, c="red")
        plt.title(key)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
    plt.show()

    lines = []
    plt.figure(figsize=(10, 8))
    plt.title("mAP")
    for key, value in maps.items():
        lines.extend(plt.plot(value, label="Segmentation" if key == "segm_map" else "Bounding Box"),)
        plt.plot(value.idxmax(), value.max(), "ro")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
    plt.legend(handles=lines)
    plt.show()
