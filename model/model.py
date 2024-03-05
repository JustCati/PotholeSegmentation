import os
import json
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model.utils import train_one_epoch_MASK, train_one_epoch_UNET




def getModel(modelType = "MASK", pretrained = True, weights = "DEFAULT", backbone_weights = "DEFAULT", device = None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = None
    if modelType == "UNET":
        model =  torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels = 3, out_channels = 1, init_features = 32, pretrained = pretrained)
        return model.to(device)

    if modelType == "MASK":
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

        return model


def trainModel(model, loader, optimizer, lr_scheduler, n_epoch, loss_func = None, path = os.getcwd(), device = None):
    losses = []
    best_loss = float("inf")

    modelType = "MASK" if isinstance(model, torchvision.models.detection.mask_rcnn.MaskRCNN) else "UNET"
    model.train()
    for epoch in range(n_epoch):
        if modelType == "MASK":
            ep_loss = train_one_epoch_MASK(model, loader, optimizer, lr_scheduler, device)
        else:
            ep_loss = train_one_epoch_UNET(model, loss_func, loader, optimizer, lr_scheduler, device)

        loss = ep_loss["total_loss"]
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(path, f"model_{modelType}.pth"))

        print(f"Epoch {epoch + 1} Total Loss: {loss}")
        losses.append((epoch, ep_loss))

    model.load_state_dict(torch.load(os.path.join(path, f"model_{modelType}.pth")))
    with open(os.path.join(path, f"losses_{modelType}.json"), "w") as f:
        json.dump(losses, f)
    return model, losses
