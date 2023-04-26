import sys
import os
import torch
import torch.nn as nn

from cocodata import get_data
from models import load_model, inference
from tools import plot_all, get_classes
from training import train_model

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\nSegmentation project running on", device)

    # training parameters
    train = False
    in_size = (520, 520)
    b_size = 2

    # model selection
    model_choice = "dlab_large"
    ft = True
    appendix = "_ft" if ft else ""

    if model_choice not in ["dlab", "dlab_large", "fcn"]:
        print("Error (wrong choice) : choose between dlab, dlab_large, or fcn")
        sys.exit(1)

    if train:

        train_ds, val_ds, test_ds, cats = get_data(input_size=in_size, batch_size=b_size, sup=True)
        nb_classes = len(cats)

        model, params_to_update = load_model(choice=model_choice, train=train, feat_extract=ft, nb_class=nb_classes)

        lr = 1e-4
        nb_epoch = 3

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(params_to_update, lr=lr)

        dls = {"train": train_ds, "val": val_ds}

        model, history = train_model(model, dls, criterion, optimizer, nb_classes, device, epochs=nb_epoch)

        torch.save(model, f"pytorch_models/{model_choice}/{model_choice}{appendix}.pt")

        plot_all(history)
    else:

        _, _, test_ds, cats = get_data(input_size=in_size, batch_size=None, sup=True)

        # load trained model if available
        m_path = f"pytorch_models/{model_choice}/{model_choice}{appendix}.pt"

        if os.path.exists(m_path):
            print("Model file found, using pretrained model for inference\n")
            nb_classes = len(cats)
            model = torch.load(m_path)
        else:
            print("Model file not found, using Pytorch's model for inference\n")
            cats = get_classes("pascal.txt")
            nb_classes = len(cats)
            model, _ = load_model(choice=model_choice)

        inference(model, test_ds, cats, nb_classes, device, nbinf=5)

    print("\nEnd of the program")
