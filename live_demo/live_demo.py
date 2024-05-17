import os
import torch
import cv2 as cv
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse

from libs.model.model import getModel, loadCheckpoint
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes







parser = argparse.ArgumentParser(description='Live demo of Mask R-CNN')
parser.add_argument("imageFolder", type=str, help="Folder with images to be processed")
args = parser.parse_args()

def getDevice():
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

device = getDevice()
model = getModel(pretrained = True, device = device)

modelOutputPath = 'mask_rcnn_50_epochs'
model, _, _, epoch = loadCheckpoint(model, path = modelOutputPath, device = device, best = True)
print(f"Loaded best model at epoch {epoch + 1}")


for imgPath in os.listdir(args.imageFolder):
    print(f"Processing image {os.path.join(os.getcwd(), args.imageFolder, imgPath)}")
    im = cv.imread(os.path.join(os.getcwd(), args.imageFolder, imgPath))
    im = cv.resize(im, (640, 640), interpolation=cv.INTER_AREA)
    img = transforms.ToTensor()(im)
    
    model.eval()
    with torch.no_grad():
        pred = model([img.to(device)])
        pred = {k: v.to("cpu") for k, v in pred[0].items()}

    for i in range(len(pred["masks"])):
        pred["masks"][i] = pred["masks"][i] > 0.7
    for i in range(len(pred['boxes'])):
        pred['scores'][i] = pred['scores'][i] > 0.7
    pred["masks"] = pred["masks"].type(torch.uint8).reshape(-1, 640, 640)


    plt.subplot(1, 2, 1)
    plt.imshow(im)

    plt.subplot(1, 2, 2)
    img = (img * 255).type(torch.uint8)
    masks = pred["masks"].type(torch.bool).reshape(-1, img.shape[-1], img.shape[-1])
    img = draw_bounding_boxes(img, pred["boxes"], colors="red", width=3)
    img = draw_segmentation_masks(img.type(torch.uint8), masks, alpha=0.5, colors="red")
    plt.imshow(transforms.ToPILImage()(img), aspect='auto')
    plt.axis('off')
    plt.show()

