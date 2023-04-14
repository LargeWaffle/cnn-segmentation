import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colormap = (colors % 255).numpy().astype("uint8")
"""
label_colors = np.array([(0, 0, 0),  # 0=background
                         # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                         (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                         # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                         (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                         # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                         (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                         # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                         (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

"""


# Define the helper function
def decode_segmap(image, nc=21):

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = colormap[l][0]
        g[idx] = colormap[l][1]
        b[idx] = colormap[l][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def image_overlay(image, segmented_image):
    alpha = 1  # transparency for the original image
    beta = 0.75  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum

    image = np.array(image)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)

    return image


def segment_map(output, img):
    om = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

    segmented_image = decode_segmap(om)

    # Resize to original image size
    segmented_image = cv2.resize(segmented_image, img.size, cv2.INTER_CUBIC)
    overlayed_image = image_overlay(img, segmented_image)

    # Plot
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
    plt.imshow(overlayed_image[:, :, ::-1])

    plt.show()
