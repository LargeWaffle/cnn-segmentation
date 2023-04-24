import torch
import torch.nn as nn

from cocodata import get_data
from models import load_model, inference
from plotters import plot_all
from training import train_model


def get_classes(fpath):
    # Initialize dictionary
    label_dict = {}

    # Open file and read each line
    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            label_id, label_name = line.split(': ')

            label_dict[int(label_id)] = label_name

    return label_dict


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\nSegmentation project running on", device)

    train = True
    in_size = (520, 520)
    b_size = 2

    model_choice = "dlab"

    if train:

        train_ds, val_ds, test_ds, nb_classes = get_data(input_size=in_size, batch_size=b_size, sup=True)

        model, params_to_update = load_model(choice=model_choice, train=train, feat_extract=True, nb_class=nb_classes)

        lr = 1e-4
        nb_epoch = 2

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        dls = {"train": train_ds, "val": val_ds}

        model, history = train_model(model, dls, criterion, optimizer, nb_classes, device, epochs=nb_epoch)

        plot_all(history)
    else:
        class_list = get_classes("pascal.txt")
        nb_classes = len(class_list)

        _, _, test_ds, _ = get_data(input_size=in_size, batch_size=b_size)

        model, _ = load_model(choice=model_choice)

        inference(model, test_ds, class_list, nb_classes, device, nbinf=5)

    print("\nEnd of the program")

    """
    from gui import App

    app = App(appw=1400, appy=600)
    app.mainloop()
    """
