import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP



def evaluate_one_epoch(model, valLoader, MASK_THRESHOLD, device):
    total_val = []
    with torch.no_grad():
        model.eval()
        for images, targets in valLoader:
            images = list([image.to(device) for image in images])
            targets = [{k: v.to(device) for k, v in elem.items()} for elem in targets]
            imgSize_X = images[0].shape[-1]
            imgSize_Y = images[0].shape[-2]

            pred = model(images)
            pred = [{k: (v > MASK_THRESHOLD).reshape(-1 , imgSize_X, imgSize_Y) if k == "masks" else v for k, v in elem.items()} for elem in pred]
            targets = [{k: (v > MASK_THRESHOLD).reshape(-1 , imgSize_X, imgSize_Y) if k == "masks" else v for k, v in elem.items()} for elem in targets]

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
    return {k: sum(acc[k] for acc in total_val) / len(total_val) for k in total_val[0]}
