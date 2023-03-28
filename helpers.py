import random

import cv2
import numpy as np
import skimage.io as skio
from pycocotools.coco import COCO
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def filter_dataset(folder, classes=None, mode='train'):
    # initialize COCO api for instance annotations
    ann_file = '{}/annotation_folder/annotations/instances_{}2017.json'.format(folder, mode)

    coco = COCO(ann_file)

    images = []

    if classes:
        img_ids = None
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            cat_ids = coco.getCatIds(catNms=className)
            img_ids = coco.getImgIds(catIds=cat_ids)

            images += coco.loadImgs(img_ids)
    else:
        img_ids = coco.getImgIds()
        images = coco.loadImgs(img_ids)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    random.shuffle(unique_images)
    dataset_size = len(unique_images)

    return unique_images, dataset_size, coco, img_ids


def count_filtered(coco, classes):
    images = []
    if classes is not None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given class
            cat_ids = coco.getCatIds(catNms=className)
            img_ids = coco.getImgIds(catIds=cat_ids)
            images += coco.loadImgs(img_ids)
    else:
        img_ids = coco.getImgIds()
        images = coco.loadImgs(img_ids)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    dataset_size = len(unique_images)

    print("Number of images containing the filter classes:", dataset_size)


def get_class_name(class_id, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == class_id:
            return cats[i]['name']
    return "None"


def format_img(image_obj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = skio.imread(img_folder + '/' + image_obj['file_name']) / 255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if len(train_img.shape) == 3 and train_img.shape[2] == 3:  # If it is a RGB 3 channel image
        return train_img
    else:  # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,) * 3, axis=-1)
        return stacked_img


def get_normal_mask(image_obj, classes, coco, cat_ids, input_image_size):

    ann_ids = coco.getAnnIds(image_obj['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    cats = coco.loadCats(cat_ids)

    train_mask = np.zeros(input_image_size)

    for a in range(len(anns)):
        class_name = get_class_name(anns[a]['category_id'], cats)

        pixel_value = classes.index(class_name) + 1

        new_mask = cv2.resize(coco.annToMask(anns[a]) * pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1])
    return train_mask


def get_binary_mask(image_obj, coco, cat_ids, input_image_size):
    ann_ids = coco.getAnnIds(image_obj['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)

        # Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1])
    return train_mask


def data_gen_coco(images, classes, coco, folder,
                  input_image_size=(224, 224), batch_size=4, mode='train', mask_type='binary'):

    img_folder = '{}/images/{}'.format(folder, mode)
    dataset_size = len(images)
    cat_ids = coco.getCatIds(catNms=classes)

    c = 0
    while True:
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')

        for i in range(c, c + batch_size):  # initially from 0 to batch_size, when c = 0
            image_obj = images[i]

            # retrieve Image
            train_img = format_img(image_obj, img_folder, input_image_size)

            # create Mask
            if mask_type == "binary":
                train_mask = get_binary_mask(image_obj, coco, cat_ids, input_image_size)

            else:  # replaces mask_type == "normal"
                train_mask = get_normal_mask(image_obj, classes, coco, cat_ids, input_image_size)

                # Add to respective batch sized arrays
            img[i - c] = train_img
            mask[i - c] = train_mask

        c += batch_size
        if c + batch_size >= dataset_size:
            c = 0
            random.shuffle(images)
        yield img, mask


def data_test_gen_coco(images, folder, batch_size, input_image_size=(224, 224)):

    img_folder = '{}/images/{}'.format(folder, 'test')
    dataset_size = len(images)

    c = 0
    while True:
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')

        for i in range(c, c + batch_size):  # initially from 0 to batch_size, when c = 0

            image_obj = images[i]

            # retrieve Image
            train_img = format_img(image_obj, img_folder, input_image_size)

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
    for i, img in enumerate(predictions):
        show_seg_img(img)
        if i > 4:
            break


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
