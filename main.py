import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random

from pycocotools.coco import COCO
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory

import augmentation as aug
import helpers as tools
from cnn import get_model, get_premade_model, train_model, predict_images

nb_epochs = 10
batch_size = 5
test_ds_size = 0.2
input_image_size = (224, 224)  # arbitrary, downsize every img
mask_type = 'normal'  # normal or binary

if __name__ == "__main__":
    data_folder = 'data'
    annFile = '{}/annotation_folder/annotations/instances_{}2017.json'.format(data_folder, 'train')

    coco = COCO(annFile)

    print("### Loading and selecting categories ###")
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    desired_classes = random.sample(nms, 10)  # nb of class we take
    print("Chosen classes:", *desired_classes)

    print("### Loading data ###")
    test_gen = image_dataset_from_directory('./data/images', labels=None, label_mode=None, batch_size=batch_size,
                                            image_size=input_image_size, validation_split=test_ds_size,
                                            seed=42, subset='validation')

    del coco

    manual = False
    if manual:
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

        train_aug, val_aug = aug.augment_data(train_gen, val_gen)

        model = get_model(input_image_size, len(desired_classes), before_pooling=1, nb_pooling=4)

        history = train_model(model=model, train_data=train_aug, val_data=val_aug,
                              steps=epoch_steps, val_steps=val_steps, epochs=nb_epochs)

    else:
        model = get_premade_model()

    # model.summary(expand_nested=True)

    # predictions will have shape (batch_size, h, w, dataset_output_classes)
    predictions = predict_images(model, test_gen)

    tools.preview_results(predictions)

    print("\n### End of program ###\n")
