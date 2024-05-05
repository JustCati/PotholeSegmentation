import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP



def evaluate_one_epoch(model, loader, MASK_THRESHOLD, tb_writer: SummaryWriter, epoch, device):
    total_val = []
    num_iters = len(loader)

    with torch.no_grad():
        model.eval()
        for iter, target in enumerate(loader):
            images, targets = target

            #* --------------- Forward Pass ----------------
            images = list([image.to(device) for image in images])
            targets = [{k: v.to(device) for k, v in elem.items()} for elem in targets]
            pred = model(images)

            #* --------------- Thresholding for binary mask ----------------
            imgSize_X = images[0].shape[-1]
            imgSize_Y = images[0].shape[-2]
            pred = [{k: (v > MASK_THRESHOLD).reshape(-1 , imgSize_X, imgSize_Y) if k == "masks" else v for k, v in elem.items()} for elem in pred]
            targets = [{k: (v > MASK_THRESHOLD).reshape(-1 , imgSize_X, imgSize_Y) if k == "masks" else v for k, v in elem.items()} for elem in targets]

            #* --------------- Compute mAP ----------------
            map_segm = MAP(box_format="xyxy", iou_type="segm")
            map_bbox = MAP(box_format="xyxy", iou_type="bbox")
            map_segm.update(pred, targets)
            map_bbox.update(pred, targets)

            res_segm = {"segm_" + k: v.item() for k, v in map_segm.compute().items()}
            res_bbox = {"bbox_" + k: v.item() for k, v in map_bbox.compute().items()}
            acc = {"segm_map": res_segm["segm_map"], "bbox_map": res_bbox["bbox_map"]}
            total_val.append(acc)

            #* --------------- Log Progress ----------------
            if iter % 10 == 0 or iter == num_iters - 1:
                print('[Validation] Epoch: [{:03d}][{:05d}/{:05d}]'.format(epoch, iter, num_iters))

        #* --------------- Log mAP ----------------
        final_acc = {k: sum(acc[k] for acc in total_val) / len(total_val) for k in total_val[0]}
        tb_writer.add_scalars("val/map", final_acc, epoch)
        print("[Validation] Epoch: [{:03d}] Segmentation mAP: {:.2f}, Bounding Box mAP: {:.2f}".format(epoch, final_acc["segm_map"], final_acc["bbox_map"]))
    return


            val_acc = {}
            val_acc.update(res_segm)
            val_acc.update(res_bbox)

            total_val.append(val_acc)
    return {k: sum(acc[k] for acc in total_val) / len(total_val) for k in total_val[0]}
