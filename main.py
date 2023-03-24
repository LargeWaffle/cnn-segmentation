import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

import augmentation as aug
import helpers as tools
from cnn import get_premade_model, preview_results
"""
data_folder = 'data'
annFile = '{}/annotation_folder/annotations/instances_{}2017.json'.format(data_folder, 'train')

coco = COCO(annFile)

print("### Loading and selecting categories ###")
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
desired_classes = random.sample(nms, 5)  # nb of class we take
# desired_classes = ['laptop', 'tv', 'cell phone']

print("### Loading data ###")
train_img, train_size, coco_train, train_img_ids = tools.filterDataset(data_folder, desired_classes, 'train')
val_img, val_size, coco_val, _ = tools.filterDataset(data_folder, desired_classes, 'val')
test_img, test_size, coco_test, test_img_ids = tools.filterDataset(data_folder, desired_classes, 'val')

nb_epochs = 10
batch_size = 5
input_image_size = (224, 224)  # arbitrary, downsize every img
mask_type = 'normal'  # normal or binary

print("### Creating data generators ###")
train_gen = tools.dataGeneratorCoco(train_img, desired_classes, coco_train, data_folder,
                                    mode='train', batch_size=batch_size, mask_type=mask_type)

val_gen = tools.dataGeneratorCoco(val_img, desired_classes, coco_val, data_folder,
                                  mode='val', batch_size=batch_size, mask_type=mask_type)


epoch_steps = train_size // batch_size
val_steps = val_size // batch_size

augGeneratorArgs = dict(featurewise_center=False,
                        samplewise_center=False,
                        rotation_range=5,
                        width_shift_range=0.01,
                        height_shift_range=0.01,
                        brightness_range=(0.8, 1.2),
                        shear_range=0.01,
                        zoom_range=[1, 1.25],
                        horizontal_flip=True,
                        vertical_flip=False,
                        fill_mode='reflect',
                        data_format='channels_last')

print("### Data augmentation ###")
train_aug = aug.augmentationsGenerator(train_gen, augGeneratorArgs)
val_aug = aug.augmentationsGenerator(val_gen, augGeneratorArgs)

aug.visualizeGenerator(train_aug)
"""

model = get_premade_model()
# model.summary(expand_nested=True)

test = tools.load_image_into_numpy_array("data/images/test/000000000001.jpg")
plt.imshow(test[0])
plt.show()

# Predictions will have shape (batch_size, h, w, dataset_output_classes)
predictions = model.predict(test)
features = model.get_features(test)
pred = predictions[0]


print("\n### End of program ###\n")
