import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


def plot_all(history):
    plot_metric(history, "acc")
    plot_metric(history, "loss")
    plot_metric(history, "score")


def plot_metric(data, lb):
    plt.plot(data[lb]['train'], label=f'train_{lb}', marker='o')
    plt.plot(data[lb]['val'], label=f'val_{lb}', marker='o')
    plt.title(f'{lb} per epoch')
    plt.ylabel(lb)
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_results(img, segmented_image, overlayed_image):

    plt.figure(figsize=(12, 5), dpi=100)
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("Image")
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title("Segmentation")
    plt.axis("off")
    plt.imshow(segmented_image)

    plt.subplot(1, 3, 3)
    plt.title("Overlayed")
    plt.axis("off")
    plt.imshow(overlayed_image)

    plt.show()
