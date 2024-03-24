import os
import json
import torch
import shutil

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP




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




def train_one_epoch(model, loader, optimizer, lr_scheduler, device):
    total_losses = []

    for images, targets in loader:
        images = list([image.to(device) for image in images])
        targets = [{k: v.to(device) for k, v in elem.items()} for elem in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        losses = {k: v.item() for k, v in loss_dict.items()}
        losses["total_loss"] = loss.item() / len(images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_losses.append(losses)
    return {k: sum(loss[k] for loss in total_losses) / len(total_losses) for k in total_losses[0]}




def trainModel(model, trainLoader, valLoader, optimizer, lr_scheduler, n_epoch, path = os.getcwd(), device = torch.device("cpu")):
    val_accuracy = {}
    train_losses = {}
    best_Acc = float("-inf")

    for epoch in range(n_epoch):

        #* --------------- Train the model ----------------
        model.train()
        train_loss = train_one_epoch(model, trainLoader, optimizer, lr_scheduler, device)
        train_losses.update({int(epoch): train_loss})

        #* --------------- Evaluate the model -------------
        total_val = []
        model.eval()
        with torch.no_grad():
            for images, targets in valLoader:
                images = list([image.to(device) for image in images])
                targets = [{k: v.to(device) for k, v in elem.items()} for elem in targets]

                pred = model(images)
                pred = [{k: (v > 0).reshape(-1 , 640, 640) if k == "masks" else v for k, v in elem.items()} for elem in pred]
                targets = [{k: (v > 0).reshape(-1 , 640, 640) if k == "masks" else v for k, v in elem.items()} for elem in targets]

                map_segm = MAP(box_format="xyxy", iou_type="segm")
                map_bbox = MAP(box_format="xyxy", iou_type="bbox")

                map_segm.update(pred, targets)
                map_bbox.update(pred, targets)

                res_segm = {"segm_" + k: v.item() for k, v in map_segm.compute().items()}
                res_bbox = {"bbox_" + k: v.item() for k, v in map_bbox.compute().items()}

                val_acc = {}
                val_acc.update(res_segm)
                val_acc.update(res_bbox)

                total_val.append(val_acc)
        #* ------------------------------------------------

        val_accuracy.update({int(epoch): {k: sum(acc[k] for acc in total_val) / len(total_val) for k in total_val[0]}})

        output = f"Epoch {epoch + 1}: \n"
        output += f"Training Total Loss: {train_loss['total_loss']},\n"
        output += f"Training Mask Loss: {train_loss['loss_mask']},\n"
        output += f"Training Box Loss: {train_loss['loss_box_reg']},\n"
        output += f"Validation Segmentation Accuracy (mAP): {val_acc['segm_map']},\n"
        output += f"Validation Box Accuracy (mAP): {val_acc['bbox_map']}\n"
        print(output)

        saveCheckpoint(model, optimizer, lr_scheduler, epoch, (val_accuracy[epoch]["segm_map"] > best_Acc), path = path)
        if val_accuracy[epoch]["segm_map"] > best_Acc:
            best_Acc = val_accuracy[epoch]["segm_map"]

    model, *_ = loadCheckpoint(model, optimizer, lr_scheduler, path = path, device = device)
    with open(os.path.join(path, "TrainLosses.json"), "w") as f:
        json.dump(train_losses, f)
    with open(os.path.join(path, "ValAccuracy.json"), "w") as f:
        json.dump(val_accuracy, f)
    return model.to(device)
