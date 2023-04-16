import numpy as np
import torch


def pixel_accuracy(output, mask):
    correct = torch.eq(output, mask).int()

    accuracy = correct.sum(dim=(1, 2, 3)) / (mask.shape[2] * mask.shape[3])
    mean_accuracy = accuracy.mean()

    return mean_accuracy.item()


def iou(pred, target, n_classes=3):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union > 0:
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious)


def mIoU(pred_mask, mask, n_classes, smooth=1e-10):
    pred_mask = pred_mask.view(-1)
    mask = mask.view(-1)

    iou_per_class = []
    for clas in range(n_classes):
        true_class = pred_mask == clas
        true_label = mask == clas

        intersect = (true_class & true_label).float().sum()
        union = (true_class | true_label).float().sum()

        if union == 0:
            iou_per_class.append(np.nan)
        else:
            iou_val = (intersect + smooth) / (union + smooth)
            iou_per_class.append(iou_val)

    return np.nanmean(iou_per_class)
