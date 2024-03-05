import os
import json
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor




def getModel(pretrained = True, weights = "DEFAULT", backbone_weights = "DEFAULT", device = None):
    if device is None:
        device = torch.device('cpu')

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

    return model




def train_one_epoch(model, loader, optimizer, lr_scheduler, device):
    total_losses = []
    for images, targets in loader:
        images = list([image.to(device) for image in images])
        targets = [{k: v.to(device) for k, v in elem.items()} for elem in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        losses = {k: v.item() for k, v in loss_dict.items()}
        losses["total_loss"] = loss.item() / len(loss_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_losses.append(losses)
    return {k: sum(loss[k] for loss in total_losses) / len(total_losses) for k in total_losses[0]}




def trainModel(model, loader, optimizer, lr_scheduler, n_epoch, path = os.getcwd(), device = None):
    losses = []
    best_loss = float("inf")

    model.train()
    for epoch in range(n_epoch):
        ep_loss = train_one_epoch(model, loader, optimizer, lr_scheduler, device)

        loss = ep_loss["total_loss"]
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(path, "model.pth"))

        print(f"Epoch {epoch + 1} Total Loss: {loss}")
        losses.append((epoch, ep_loss))

    model.load_state_dict(torch.load(os.path.join(path, "model.pth")))
    with open(os.path.join(path, "losses.json"), "w") as f:
        json.dump(losses, f)
    return model, losses
