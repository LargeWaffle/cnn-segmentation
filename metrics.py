import numpy as np
import torch


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


def computeMetrics(confusion):
    # Init
    label_count = confusion.shape[0]
    ious = np.zeros(label_count)
    maccs = np.zeros(label_count)
    ious[:] = np.NAN
    maccs[:] = np.NAN

    # Get true positives, positive predictions and positive ground-truth
    total = confusion.sum()
    if total <= 0:
        raise Exception('Error: Confusion matrix is empty!')
    tp = np.diagonal(confusion)
    pos_pred = confusion.sum(axis=0)
    posGt = confusion.sum(axis=1)

    # Check which classes have elements
    valid = posGt > 0
    ious_valid = np.logical_and(valid, posGt + pos_pred - tp > 0)

    # Compute per-class results and frequencies
    ious[ious_valid] = np.divide(tp[ious_valid], posGt[ious_valid] + pos_pred[ious_valid] - tp[ious_valid])
    maccs[valid] = np.divide(tp[valid], posGt[valid])
    freqs = np.divide(posGt, total)

    # Compute evaluation metrics
    miou = np.mean(ious[ious_valid])
    fwiou = np.sum(np.multiply(ious[ious_valid], freqs[ious_valid]))
    macc = np.mean(maccs[valid])
    pacc = tp.sum() / total

    return miou, fwiou, macc, pacc, ious,


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
