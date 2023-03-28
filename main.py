import random

import tensorflow as tf
from keras.utils import image_dataset_from_directory
from pycocotools.coco import COCO

import augmentation as aug
import helpers as tools
from cnn import get_model, get_premade_model, train_model

nb_epochs = 10
batch_size = 5
test_split = 0.0005
input_image_size = (224, 224)  # arbitrary, downsize every img
mask_type = 'normal'  # normal or binary

if __name__ == "__main__":
    data_folder = 'data'

    print("### Loading data ###")
    test_dataset = image_dataset_from_directory("data/images/folder", labels=None, label_mode=None, batch_size=None,
                                                image_size=input_image_size, validation_split=test_split,
                                                subset='validation', seed=random.randint(0, 50))

    manual = False
    if manual:
        ann_file = '{}/annotation_folder/annotations/instances_{}2017.json'.format(data_folder, 'train')
        coco = COCO(ann_file)

        print("### Loading and selecting categories ###")
        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        desired_classes = random.sample(nms, 10)  # nb of class we take
        print("Chosen classes:", *desired_classes)

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

    predictions = test_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(model.predict([x])))

    tools.preview_results(test_dataset, predictions, input_image_size)

    print("\n### End of program ###\n")
