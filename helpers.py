import random

import cv2
import numpy as np
import skimage.io as skio
from pycocotools.coco import COCO
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def filterDataset(folder, classes=None, mode='train'):
    # initialize COCO api for instance annotations
    if mode == 'test':
        annFile = '{}/annotation_folder/annotations/image_info_{}2017.json'.format(folder, mode)
    else:
        annFile = '{}/annotation_folder/annotations/instances_{}2017.json'.format(folder, mode)

    coco = COCO(annFile)

    images = []

    if classes:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)

    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    random.shuffle(unique_images)
    dataset_size = len(unique_images)

    return unique_images, dataset_size, coco, imgIds


def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return None


def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = skio.imread(img_folder + '/' + imageObj['file_name']) / 255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if len(train_img.shape) == 3 and train_img.shape[2] == 3:  # If it is a RGB 3 channel image
        return train_img
    else:  # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,) * 3, axis=-1)
        return stacked_img


def getNormalMask(imageObj, classes, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], cats)
        pixel_value = classes.index(className) + 1
        new_mask = cv2.resize(coco.annToMask(anns[a]) * pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def getBinaryMask(imageObj, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)

        # Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def dataGeneratorCoco(images, classes, coco, folder,
                      input_image_size=(224, 224), batch_size=4, mode='train', mask_type='binary'):
    img_folder = '{}/images/{}'.format(folder, mode)
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)

    c = 0
    while True:
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')

        for i in range(c, c + batch_size):  # initially from 0 to batch_size, when c = 0
            imageObj = images[i]

            # retrieve Image
            train_img = getImage(imageObj, img_folder, input_image_size)

            # create Mask
            if mask_type == "binary":
                train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size)

            else:  # replaces mask_type == "normal"
                train_mask = getNormalMask(imageObj, classes, coco, catIds, input_image_size)

                # Add to respective batch sized arrays
            img[i - c] = train_img
            mask[i - c] = train_mask

        c += batch_size
        if c + batch_size >= dataset_size:
            c = 0
            random.shuffle(images)
        yield img, mask


def dataTestGeneratorCoco(images, folder, input_image_size=(224, 224), batch_size=4):
    img_folder = '{}/images/{}'.format(folder, 'test')
    dataset_size = len(images)

    c = 0
    while True:
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')

        for i in range(c, c + batch_size):  # initially from 0 to batch_size, when c = 0
            imageObj = images[i]

            # retrieve Image
            train_img = getImage(imageObj, img_folder, input_image_size)

            # Add to respective batch sized arrays
            img[i - c] = train_img

        c += batch_size
        if c + batch_size >= dataset_size:
            c = 0
            random.shuffle(images)
        yield img


"""
def preview_results(img_list, feats=None):
    num_images = len(img_list)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    for i, img in enumerate(img_list):
        axes[i].imshow(img)
        axes[i].set_title("")

        if feats is not None:
            axes[i].imshow(feats)
            axes[i].set_title("")

    plt.tight_layout()
    plt.show()
"""

color_list = [mcolors.to_rgb(hexcolor) for hexcolor in mcolors.CSS4_COLORS.values()]


def preview_results(predictions, features=None):
    for img in predictions:
        show_seg_img(img)


def show_seg_img(img):
    pred = img.numpy().reshape(-1, 202)
    pred_classes = np.argmax(pred, axis=1)

    pred_classes = pred_classes.reshape(480, 640)

    segmentation_image = np.zeros((480, 640, 3), dtype=np.float32)
    for i in range(480):
        for j in range(640):
            class_index = pred_classes[i, j] % len(color_list)
            segmentation_image[i, j, :] = color_list[class_index]

    plt.imshow(segmentation_image)
    plt.show()
    plt.show()
