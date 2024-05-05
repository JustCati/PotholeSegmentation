import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter

from .evaluate import evaluate_one_epoch
from .model import saveCheckpoint, loadCheckpoint



def train_one_epoch(model, loader, optimizer, lr_scheduler, tb_writer: SummaryWriter, epoch, device):
    num_iters = len(loader)

    model.train()
    for iter, target in enumerate(loader):
        #* --------------- Forward Pass ----------------
        images, targets = target
        images = list([image.to(device) for image in images])
        targets = [{k: v.to(device) for k, v in elem.items()} for elem in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        #* --------------- Log Losses ----------------
        global_step = epoch * num_iters + iter

        for k, v in loss_dict.items():
            tb_writer.add_scalar(f"train/{k}", v, global_step)
        tb_writer.add_scalar("train/total_loss", loss.item() / len(images), global_step)

        #* --------------- Backward and Optimize ----------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        #* --------------- Log Learning Rate ----------------
        tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        #* --------------- Log Progress ----------------

        block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]  Loss: {:.4f} \n'.format(epoch, iter, num_iters, loss.item() / len(images))
        block2 = 'Box Loss: {:.4f}'.format(loss_dict['loss_box_reg'].item())
        block3 = "Mask Loss: {:.4f}".format(loss_dict['loss_mask'].item())

        if iter % 10 == 0 or iter == num_iters - 1:
            output = "\t".join([block1, block2, block3])
            print(output)
    print("[Train] Epoch: [{:03d}] Loss: {:.4f} Lr: {:.8f}".format(epoch, loss.item() / len(images), optimizer.param_groups[0]["lr"]))
    return



def trainModel(cfg):

    #* --------------- Load Config ----------------
    model = cfg["model"]
    optimizer = cfg["optimizer"]
    lr_scheduler = cfg["lr_scheduler"]
    last_epoch = cfg["last_epoch"]
    n_epoch = cfg["epoch"]
    MASK_THRESHOLD = cfg["mask_threshold"]
    device = cfg["device"]
    trainLoader = cfg["trainDataloader"]
    valLoader = cfg["valDataloader"]
    path = cfg["path"]
    tb_writer = cfg["tb_writer"]
    #* --------------------------------------------


    val_accuracy = {}
    train_losses = {}
    best_Acc = float("-inf")
    last_epoch = last_epoch + 1 if last_epoch != 0 else 0

    for epoch in range(n_epoch):

        #* --------------- Train the model ----------------

        model.train()
        train_loss = train_one_epoch(model, trainLoader, optimizer, lr_scheduler, device)
        train_losses.update({int(last_epoch + epoch + 1): train_loss})

        #* --------------- Evaluate the model -------------

        model.eval()
        train_acc = evaluate_one_epoch(model, valLoader, MASK_THRESHOLD, device)
        val_accuracy.update({int(last_epoch + epoch + 1): train_acc})

        #* ------------------------------------------------


        output = f"Epoch {last_epoch + epoch + 1}: \n"
        output += f"Training Total Loss: {train_loss['total_loss']:.2f},\n"
        output += f"Training Mask Loss: {train_loss['loss_mask']:.2f},\n"
        output += f"Training Box Loss: {train_loss['loss_box_reg']:.2f},\n"
        output += f"Validation Segmentation Accuracy (mAP): {train_acc['segm_map']:.2f},\n"
        output += f"Validation Box Accuracy (mAP): {train_acc['bbox_map']:.2f}\n"
        print(output)

        saveCheckpoint(model, optimizer, lr_scheduler, last_epoch + epoch, (val_accuracy[last_epoch + epoch + 1]["segm_map"] > best_Acc), path = path)
        if val_accuracy[last_epoch + epoch + 1]["segm_map"] > best_Acc:
            print(f"Best model found at epoch {last_epoch + epoch + 1} with mAP: {val_accuracy[last_epoch + epoch + 1]['segm_map']:.2f}, saving....")
            best_Acc = val_accuracy[last_epoch + epoch + 1]["segm_map"]

    model, *_ = loadCheckpoint(model, path = path, device = device)
    with open(os.path.join(path, "TrainLosses.json"), "a") as f:
        json.dump(train_losses, f)
    with open(os.path.join(path, "ValAccuracy.json"), "a") as f:
        json.dump(val_accuracy, f)
    return model.to(device)
