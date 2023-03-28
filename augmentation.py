import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def augment_generator(gen, aug_args, seed=None):
    # Initialize the image data generator with args provided
    image_gen = ImageDataGenerator(**aug_args)

    # Remove the brightness argument for the mask. Spatial arguments similar to image.
    aug_args_mask = aug_args.copy()
    _ = aug_args_mask.pop('brightness_range', None)

    # Initialize the mask data generator with modified args
    mask_gen = ImageDataGenerator(**aug_args_mask)

    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))

    for img, mask in gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation of the images
        # will end up different from the augmentation of the masks
        g_x = image_gen.flow(255 * img,
                             batch_size=img.shape[0],
                             seed=seed,
                             shuffle=True)
        g_y = mask_gen.flow(mask,
                            batch_size=mask.shape[0],
                            seed=seed,
                            shuffle=True)

        img_aug = next(g_x) / 255.0

        mask_aug = next(g_y)

        yield img_aug, mask_aug


def augment_data(train_generator, val_generator):
    aug_args = dict(featurewise_center=False,
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
    train_augmentation = augment_generator(train_generator, aug_args)
    val_augmentation = augment_generator(val_generator, aug_args)

    # aug.visualizeGenerator(train_aug)

    return train_augmentation, val_augmentation


def visualize_generator(gen):
    # Iterate the generator to get image and mask batches
    img, mask = next(gen)

    fig = plt.figure(figsize=(20, 10))
    outer_grid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

    for i in range(2):
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_grid[i], wspace=0.05, hspace=0.05)

        for j in range(4):
            ax = plt.Subplot(fig, inner_grid[j])
            if i == 1:
                ax.imshow(mask[j][:, :, 0])
            else:
                ax.imshow(img[j])

            ax.axis('off')
            fig.add_subplot(ax)
    plt.show()
