import numpy as np
import torch
import torch.nn as nn
# from torcheval.metrics import MulticlassF1Score, MulticlassAccuracy, MulticlassAUROC
from pycocotools.cocostuffhelper import getCMap

from cocodata import get_data
from models import load_model, inference
from plotters import plot_all
from training import train_model

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\nSegmentation project running on", device)

    train = False
    colormap = (getCMap() * 255).astype(np.uint8)

    if train:

        train_ds, val_ds, test_ds, nb_classes = get_data(input_size=(520, 520), batch_size=2)

        model, params_to_update = load_model(choice="dlab", train=train, feat_extract=True, nb_class=nb_classes)

        max_lr = 1e-3
        nb_epoch = 5
        weight_decay = 1e-4

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params_to_update, lr=max_lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=nb_epoch, steps_per_epoch=len(train_ds))

        dls = {"train": train_ds, "val": val_ds}

        model, history = train_model(model, dls, criterion, optimizer, sched, nb_classes, device, epochs=nb_epoch)

        plot_all(history)
    else:
        _, _, test_ds, nb_classes = get_data(input_size=(520, 520), batch_size=2)

        model, _ = load_model(choice="dlab", train=train, feat_extract=False, nb_class=nb_classes)

        # metrics = {'accuracy': MulticlassAccuracy, 'f1_score': MulticlassF1Score, 'auroc': MulticlassAUROC}
        inference(model, test_ds, colormap, nb_classes, device, nbinf=5)

    print("End of the program")
