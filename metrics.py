import numpy as np
import torch


def _fast_hist(label_true, label_pred, n_class):
    label_true = label_true.numpy()
    label_pred = label_pred.numpy()

    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def metrics_report(label_trues, label_preds, n_class):
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
        "pixel accuracy": acc,
        "mean accuracy": acc_cls,
        "frequency weighted IoU": fwavacc,
        "mean IoU": mean_iu,
        "class IoU": cls_iu,
    }


def pixel_accuracy(output, mask):
    correct = torch.eq(output, mask).int()

    accuracy = correct.sum(dim=(1, 2, 3)) / (mask.shape[2] * mask.shape[3])
    mean_accuracy = accuracy.mean()

    return mean_accuracy.item()


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
