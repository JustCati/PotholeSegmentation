from .model import saveCheckpoint
from .evaluate import evaluate_one_epoch

from torch.utils.tensorboard import SummaryWriter


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

        losses = {
            "loss_box_reg": loss_dict["loss_box_reg"].item(), 
            "loss_mask": loss_dict["loss_mask"].item()
        }
        tb_writer.add_scalars("train/all_losses", losses, global_step)
        tb_writer.add_scalar("train/final_loss", loss.item() / len(images), global_step)

        #* --------------- Backward and Optimize ----------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        #* --------------- Log Learning Rate ----------------
        tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        #* --------------- Log Progress ----------------

        block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]  Loss: {:.4f} \n'.format(epoch + 1, iter, num_iters, loss.item() / len(images))
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
    curr_epoch = cfg["curr_epoch"]
    n_epoch = cfg["epoch"]
    MASK_THRESHOLD = cfg["mask_threshold"]
    device = cfg["device"]
    trainLoader = cfg["trainDataloader"]
    valLoader = cfg["valDataloader"]
    path = cfg["path"]
    tb_writer = cfg["tb_writer"]
    #* --------------------------------------------

    #* --------------- Train and Evaluate ----------------
    best_Acc = float("-inf")
    print("\nStart training model...")

    for epoch in range(curr_epoch, n_epoch):
        train_one_epoch(model, 
                        trainLoader,
                        optimizer,
                        lr_scheduler,
                        tb_writer,
                        epoch,
                        device)
        train_acc = evaluate_one_epoch(model,
                                       valLoader,
                                       MASK_THRESHOLD,
                                       device,
                                       tb_writer,
                                       epoch)
        train_acc = train_acc["segm_map"]

        saveCheckpoint(model, optimizer, lr_scheduler, epoch, (train_acc > best_Acc), path = path)
        if train_acc > best_Acc:
            print(f"Best model found at epoch {epoch + 1} with mAP: {train_acc:.2f}, saving....")
            print()
            best_Acc = train_acc
    #* ----------------------------------------------

    print("Training completed")
    return
