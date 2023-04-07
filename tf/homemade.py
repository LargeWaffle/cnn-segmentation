import random
import numpy as np
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import augmentation as aug
import helpers as tools
from segmentation import DeeplabV3Plus, train_model, plot_predictions
from scipy.io import loadmat
import matplotlib.colors as mcolors


nb_epochs = 2
batch_size = 5
test_split = 0.0005
input_image_size = (640, 640)  # arbitrary, downsize every img
mask_type = 'normal'  # normal or binary

data_folder = 'data'

ann_file = f'{data_folder}/annotation_folder/annotations/instances_train2017.json'
coco = COCO(ann_file)

print("### Loading and selecting categories ###")

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
desired_classes = random.sample(nms, 10)  # nb of class we take
print("Chosen classes:", *desired_classes)

print("### Loading data ###")

train_img, train_size, coco_train, train_img_ids = tools.filter_dataset(data_folder, desired_classes, 'train')
val_img, val_size, coco_val, _ = tools.filter_dataset(data_folder, desired_classes, 'val')

epoch_steps = train_size // batch_size
val_steps = val_size // batch_size

print("### Creating data generators ###")

train_gen = tools.data_gen_coco(train_img, desired_classes, coco_train, data_folder,
                                mode='train', input_image_size=input_image_size, batch_size=batch_size,
                                mask_type=mask_type)

val_gen = tools.data_gen_coco(val_img, desired_classes, coco_val, data_folder,
                              mode='val', input_image_size=input_image_size, batch_size=batch_size,
                              mask_type=mask_type)

test_dataset = image_dataset_from_directory(data_folder + "/images/folder",
                                            labels=None,
                                            label_mode=None,
                                            batch_size=None,
                                            image_size=input_image_size,
                                            validation_split=test_split,
                                            subset='validation',
                                            seed=random.randint(0, 50))


print("### Adding data augmentation ###")

train_aug, val_aug = aug.augment_data(train_gen, val_gen)

model = DeeplabV3Plus(input_image_size, len(desired_classes))

model.summary(expand_nested=True)

history = train_model(model=model, train_data=train_aug, val_data=val_aug, epochs=nb_epochs)

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.show()

print("### Making predictions ###")

predictions = test_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(model.predict([x])))

color_list = [mcolors.to_rgb(hexcolor) for hexcolor in mcolors.CSS4_COLORS.values()]

plot_predictions(predictions[:4], color_list, model=model)

print("\n### End of program ###\n")


"""
import random

import cv2
import keras
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras.applications import ResNet50
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory
from pycocotools.coco import COCO

import augmentation as aug
import helpers as tools

nb_epochs = 2
batch_size = 5
test_split = 0.0005
input_image_size = (640, 640)  # arbitrary, downsize every img
mask_type = 'normal'  # normal or binary

data_folder = 'data'

ann_file = f'{data_folder}/annotation_folder/annotations/instances_train2017.json'
coco = COCO(ann_file)

print("### Loading and selecting categories ###")

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
desired_classes = random.sample(nms, 10)  # nb of class we take
print("Chosen classes:", *desired_classes)

print("### Loading data ###")

train_img, train_size, coco_train, train_img_ids = tools.filter_dataset(data_folder, desired_classes, 'train')
val_img, val_size, coco_val, _ = tools.filter_dataset(data_folder, desired_classes, 'val')

epoch_steps = train_size // batch_size
val_steps = val_size // batch_size

print("### Creating data generators ###")

train_gen = tools.data_gen_coco(train_img, desired_classes, coco_train, data_folder,
                                mode='train', input_image_size=input_image_size, batch_size=batch_size,
                                mask_type=mask_type)

val_gen = tools.data_gen_coco(val_img, desired_classes, coco_val, data_folder,
                              mode='val', input_image_size=input_image_size, batch_size=batch_size,
                              mask_type=mask_type)

test_dataset = image_dataset_from_directory(data_folder + "/images/folder",
                                            labels=None,
                                            label_mode=None,
                                            batch_size=None,
                                            image_size=input_image_size,
                                            validation_split=test_split,
                                            subset='validation',
                                            seed=random.randint(0, 50))

print("### Adding data augmentation ###")

train_aug, val_aug = aug.augment_data(train_gen, val_gen)


def convolution_block(
        block_input,
        num_filters=256,
        kernel_size=3,
        dilation_rate=1,
        padding="same",
        use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


model = DeeplabV3Plus(image_size=input_image_size[0], num_classes=len(desired_classes))
model.summary()

loss = SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"],
)

history = model.fit(train_aug, validation_data=val_aug, epochs=nb_epochs)

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.show()


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=input_image_size)
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=input_image_size)
        image = tf.keras.applications.resnet50.preprocess_input(image)
    return image


def plot_predictions(images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib(
            [image_tensor, overlay, prediction_colormap], figsize=(18, 14)
        )


colormap = [mcolors.to_rgb(hexcolor) for hexcolor in mcolors.CSS4_COLORS.values()]

for img in test_dataset:
    plot_predictions([img], colormap, model=model)
    break
"""
