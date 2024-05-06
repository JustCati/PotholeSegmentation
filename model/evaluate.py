import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP



def evaluate_one_epoch(model, loader, MASK_THRESHOLD, device, tb_writer: SummaryWriter = None, epoch = 1):
    total_val = []
    num_iters = len(loader)

    model.eval()
    with torch.no_grad():
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

        if tb_writer is not None:
            tb_writer.add_scalars("val/map", final_acc, epoch)

        print("[Validation] Epoch: {:03d} Segmentation mAP: {:.2f}, Bounding Box mAP: {:.2f}".format(epoch, final_acc["segm_map"], final_acc["bbox_map"]))
        print()
    return final_acc


def demo(model, img, target, MASK_THRESHOLD, BBOX_THRESHOLD, device):
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        prediction = {k: v.to("cpu") for k, v in prediction[0].items()}

    #! Thresholding for visualization (PlotDemo shows only the pixels with 1)
    for i in range(len(prediction['boxes'])):
        prediction['scores'][i] = prediction['scores'][i] > BBOX_THRESHOLD
    for i in range(len(prediction["masks"])):
        prediction["masks"][i] = prediction["masks"][i] > MASK_THRESHOLD

    pred = prediction.copy()
    pred["masks"] = pred["masks"].type(torch.uint8).reshape(-1, img.shape[-1], img.shape[-1])
    target["masks"] = target["masks"].type(torch.uint8).reshape(-1, img.shape[-1], img.shape[-1])

    map_segm = MAP(box_format="xyxy", iou_type="segm")
    map_segm.update([pred], [target])
    segm_acc = map_segm.compute()

    map_bbox = MAP(box_format="xyxy", iou_type="bbox")
    map_bbox.update([pred], [target])
    bbox_acc = map_bbox.compute()
    return {
        "img": img,
        "target": target,
        "prediction": pred,
        "segm_map": segm_acc["map"],
        "bbox_map": bbox_acc["map"]
    }
