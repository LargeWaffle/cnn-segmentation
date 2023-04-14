import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

test_split = 0.0001
input_image_size = (1024, 1024)  # arbitrary, downsize every img

data_folder = 'data/images'

print("### Loading mask_rcnn/inception_resnet_v2_1024x1024 ###")
model = tf.saved_model.load("mymodels/Mask-RCNN/")
print("### Model loaded ###")

# model.summary(expand_nested=True)

# Load the image to be segmented
print("### Loading data ###")
test_dataset = image_dataset_from_directory(data_folder + "/folder", labels=None, label_mode=None, batch_size=None,
                                            image_size=input_image_size, validation_split=test_split, color_mode="rgb",
                                            subset='val', seed=random.randint(0, 50))


def create_predictions(test_ds, detection_model):
    results = []

    PATH_TO_LABELS = '../models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    for image in test_ds:
        image_np = tf.cast(image, dtype=tf.uint8).numpy()
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        model_fn = detection_model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        num_detections = int(output_dict.pop('num_detections'))

        need_detection_key = ['detection_classes', 'detection_boxes', 'detection_masks', 'detection_scores']

        output_dict = {key: output_dict[key][0, :num_detections].numpy() for key in need_detection_key}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                tf.convert_to_tensor(output_dict['detection_masks']), output_dict['detection_boxes'],
                image.shape[0], image.shape[1])

            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=5)

        results.append((tf.cast(image, dtype=tf.uint8).numpy(), Image.fromarray(image_np)))

    return results


print("### Making predictions ###")
predictions = create_predictions(test_dataset, model)

for img, pred in predictions:
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    f.add_subplot(1, 2, 2)
    plt.imshow(pred)
    plt.axis("off")
    plt.show()

print("\n### End of program ###\n")
