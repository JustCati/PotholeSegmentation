import os
import json
import torch

from torchmetrics.detection.iou import IntersectionOverUnion as IoU
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP

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




def trainModel(model, trainLoader, valLoader, optimizer, lr_scheduler, n_epoch, path = os.getcwd(), device = None):
    val_accuracy = {}
    train_losses = {}
    best_val_acc = float("-inf")

    for epoch in range(n_epoch):
        #* Train the model
        model.train()
        train_loss = train_one_epoch(model, trainLoader, optimizer, lr_scheduler, device)
        train_losses.update({int(epoch): train_loss})

        #* Evaluate the model
        total_val = []
        model.eval()
        with torch.no_grad():
            for images, targets in valLoader:
                images = list([image.to(device) for image in images])
                targets = [{k: v.to(device) for k, v in elem.items()} for elem in targets]

                pred = model(images)

                map = MAP(box_format="xyxy", iou_type="bbox")
                iou = IoU(box_format="xyxy", iou_threshold=0.5)

                map.update(pred, targets)
                iou.update(pred, targets)

                val_acc = {}
                val_acc.update(map.compute())
                val_acc.update(iou.compute())
                val_acc = {k: v.item() for k, v in val_acc.items()}

                total_val.append(val_acc)

        accuracy = val_acc["map"]
        val_accuracy.update({int(epoch): {k: sum(acc[k] for acc in total_val) / len(total_val) for k in total_val[0]}})
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            torch.save(model.state_dict(), os.path.join(path, "model.pth"))

        print(f"Epoch {epoch + 1} Training Loss: {train_loss['total_loss']}, Validation Accuracy (mAP): {val_acc['map']}")


    model.load_state_dict(torch.load(os.path.join(path, "model.pth")))
    with open(os.path.join(path, "TrainLosses.json"), "w") as f:
        json.dump(train_losses, f)
    with open(os.path.join(path, "ValAccuracy.json"), "w") as f:
        json.dump(val_accuracy, f)
    return model, train_losses, val_accuracy
