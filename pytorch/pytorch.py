import torch
import torch.nn as nn

# from torcheval.metrics import MulticlassF1Score, MulticlassAccuracy, MulticlassAUROC
from cocodata import get_data
from models import load_model, inference
from training import train_model
from plotters import plot_all
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\nSegmentation project running on", device)

    train_ds, val_ds, test_ds, nb_classes = get_data(input_size=(520, 520), batch_size=2)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(nb_classes)])[:, None] * palette
    colormap = (colors % 255).numpy().astype("uint8")

    for img, mask in train_ds:
        plt.figure(figsize=(12, 5), dpi=100)
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Image")
        plt.imshow(img[0].permute(1, 2, 0))

        plt.subplot(1, 2, 2)
        plt.title("Segmentation")
        plt.axis("off")
        plt.imshow(mask[0].permute(1, 2, 0))

        plt.show()

    ft_extract = True
    model, params_to_update = load_model(choice="dlab", train=True, nb_class=nb_classes, feat_extract=ft_extract)

    max_lr = 1e-3
    nb_epoch = 15
    weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params_to_update, lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=nb_epoch, steps_per_epoch=len(train_ds))

    dls = {"train": train_ds, "val": val_ds}

    model, history = train_model(model, dls, criterion, optimizer, sched, nb_classes, device, epochs=nb_epoch)

    plot_all(history)

    # metrics = {'accuracy': MulticlassAccuracy, 'f1_score': MulticlassF1Score, 'auroc': MulticlassAUROC}
    inference(model, test_ds, colormap, device, nbinf=5)

    print("End of the program")
