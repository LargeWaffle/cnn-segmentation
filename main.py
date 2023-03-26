import random

from pycocotools.coco import COCO

import augmentation as aug
import helpers as tools
from cnn import get_model, get_premade_model, train_model

nb_epochs = 10
batch_size = 5
input_image_size = (224, 224)  # arbitrary, downsize every img
mask_type = 'normal'  # normal or binary


def augment(train_generator, val_generator):
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
    train_augmentation = aug.augmentationsGenerator(train_generator, augGeneratorArgs)
    val_augmentation = aug.augmentationsGenerator(val_generator, augGeneratorArgs)

    # aug.visualizeGenerator(train_aug)

    return train_augmentation, val_augmentation


def premade():
    return get_premade_model()


def memade(input_size, num_classes, bp, nb_p):
    return get_model(input_size, num_classes, before_pooling=bp, nb_pooling=nb_p)


if __name__ == "__main__":
    data_folder = 'data'
    annFile = '{}/annotation_folder/annotations/instances_{}2017.json'.format(data_folder, 'train')

    coco = COCO(annFile)

    print("### Loading and selecting categories ###")
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    desired_classes = random.sample(nms, 5)  # nb of class we take
    print("Chosen classes:", *desired_classes)

    print("### Loading data ###")
    test_img, _, _, _ = tools.filterDataset(data_folder, desired_classes, 'test')
    test_gen = tools.dataTestGeneratorCoco(test_img, data_folder, input_image_size=input_image_size,
                                           batch_size=batch_size)

    manual = False
    if manual:
        train_img, train_size, coco_train, train_img_ids = tools.filterDataset(data_folder, desired_classes, 'train')
        val_img, val_size, coco_val, _ = tools.filterDataset(data_folder, desired_classes, 'val')

        epoch_steps = train_size // batch_size
        val_steps = val_size // batch_size

        print("### Creating data generators ###")
        train_gen = tools.dataGeneratorCoco(train_img, desired_classes, coco_train, data_folder,
                                            mode='train', input_image_size=input_image_size, batch_size=batch_size,
                                            mask_type=mask_type)

        val_gen = tools.dataGeneratorCoco(val_img, desired_classes, coco_val, data_folder,
                                          mode='val', input_image_size=input_image_size, batch_size=batch_size,
                                          mask_type=mask_type)

        train_aug, val_aug = augment(train_gen, val_gen)

        model = memade(input_image_size, len(desired_classes), bp=1, nb_p=4)

        history = train_model(model=model, train_data=train_aug, val_data=val_aug,
                              steps=epoch_steps, val_steps=val_steps, epochs=nb_epochs)

    else:
        model = premade()

    # model.summary(expand_nested=True)

    # Predictions will have shape (batch_size, h, w, dataset_output_classes)
    predictions = model.predict(test_gen)
    features = model.get_features(test_gen)

    tools.preview_results(predictions, features)

    print("\n### End of program ###\n")
