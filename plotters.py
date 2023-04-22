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


def plot_results(img, segmented_image, overlayed_image, cn):

    # Create the figure and subplots
    fig, axs = plt.subplots(1, 4, figsize=(12, 5), dpi=100)

    axs[0].axis("off")
    axs[0].set_title("Image")
    axs[0].imshow(img)

    axs[1].set_title("Segmentation")
    axs[1].axis("off")
    axs[1].imshow(segmented_image)

    axs[2].set_title("Overlayed")
    axs[2].axis("off")
    axs[2].imshow(overlayed_image)

    axs[3].set_title("Detected objects")
    axs[3].axis("off")
    axs[3].text(0, 0, '\n'.join(cn))

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()
