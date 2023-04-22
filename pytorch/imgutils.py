import cv2
import numpy as np
import torch


# Define the helper function
def decode_segmap(image, colormap, nc=21):
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def segment_map(output, img, colormap, nb_class):
    om = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

    segmented_image = decode_segmap(om, colormap, nb_class)

    # Resize to original image size
    segmented_image = cv2.resize(segmented_image, om.shape, cv2.INTER_CUBIC)

    np_img = np.array(img * 255, dtype=np.uint8)

    overlayed_image = image_overlay(np_img, segmented_image)

    return segmented_image, overlayed_image
