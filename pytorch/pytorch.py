import numpy as np
import torch
import torch.nn as nn
from pycocotools.cocostuffhelper import getCMap

from cocodata import get_data
from models import load_model, inference
from plotters import plot_all
from training import train_model

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\nSegmentation project running on", device)

    train = False
    colormap = (getCMap(addThings=False) * 255).astype(np.uint8)

    if train:

        train_ds, val_ds, test_ds, cats = get_data(input_size=(520, 520), batch_size=2)
        nb_classes = len(cats)

        model, params_to_update = load_model(choice="dlab", train=train, feat_extract=True, nb_class=nb_classes)

        max_lr = 1e-3
        nb_epoch = 5
        weight_decay = 1e-4

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params_to_update, lr=max_lr, weight_decay=weight_decay)

        dls = {"train": train_ds, "val": val_ds}

        model, history = train_model(model, dls, criterion, optimizer, nb_classes, device, epochs=nb_epoch)

        plot_all(history)
    else:
        _, _, test_ds, cats = get_data(input_size=(520, 520), batch_size=2)
        nb_classes = len(cats)
        model, _ = load_model(choice="dlab", train=train, feat_extract=False, nb_class=nb_classes)

        cats = np.array(cats)
        inference(model, test_ds, colormap, cats, nb_classes, device, nbinf=5)

    print("End of the program")
