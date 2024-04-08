import os
import torch
import shutil

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor




def saveCheckpoint(model, optimizer, lr_scheduler, epoch, isBest = False, path = os.getcwd()):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict()
    }, os.path.join(path, f"checkpoint.pth"))

    if isBest:
        shutil.copyfile(os.path.join(path, "checkpoint.pth"), os.path.join(path, "model.pth"))


def loadCheckpoint(model, optimizer = None, lr_scheduler = None, path = os.getcwd(), device = torch.device("cpu")):
    checkpoint = torch.load(os.path.join(path, "checkpoint.pth"))

    epoch = checkpoint["epoch"] if "epoch" in checkpoint else 0
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) if optimizer is not None else None
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"]) if lr_scheduler is not None else None
    return model.to(device), optimizer, lr_scheduler, epoch



def getModel(pretrained = True, weights = "DEFAULT", backbone_weights = "DEFAULT", device = torch.device("cpu")):
    model = None
    if pretrained:
        model = maskrcnn_resnet50_fpn_v2(weights = weights, backbone_weights = backbone_weights)
    else:
        model = maskrcnn_resnet50_fpn_v2()

    #* Change the number of output classes
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels = in_features_box, num_classes = 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels = in_features_mask, dim_reduced = dim_reduced, num_classes = 2)

    return model.to(device)
