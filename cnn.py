import numpy as np

from imports import COCOeval, Precision, Recall, AUC, plt, Image, random
from tensorflow import saved_model
import tensorflow_hub as hub


def get_premade_model():
    print("### Model from TensorFlow Hub ###")
    print("### HRNet_coco-hrnetv2-w48_1 ###")
    # model = hub.load('https://tfhub.dev/google/HRNet/coco-hrnetv2-w48/1')
    model = saved_model.load("models/HRNet/")
    print("### Model loaded ###")

    return model


def get_model():
    print("### Model compiling ###")
    # model =  # model
    # opt =  # opti
    # lossFn =  # loss func

    # model.compile(loss=lossFn, optimizer=opt, metrics=['accuracy', 'categorical_accuracy',
    # Precision(), Recall(), AUC()])
    pass
    # return model


def train_model(*, model, train_data, val_data, steps, val_steps, epochs):
    print("### Model training ###")

    # Start the training process
    history = model.fit(x=train_data,
                        validation_data=val_data,
                        steps_per_epoch=steps,
                        validation_steps=val_steps,
                        epochs=epochs,
                        verbose=True)


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


def preview_results(predictions, features):
    pass


def show_seg_img(img):
    pred = img.numpy().reshape(-1, 202)
    pred_classes = np.argmax(pred, axis=1)

    pred_classes = pred_classes.reshape(480, 640)

    color_dict = {}
    segmentation_image = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(480):
        for j in range(640):
            class_index = pred_classes[i, j]
            segmentation_image[i, j, :] = random.choices(range(256), k=3)

    # Step 6
    segmentation_image = Image.fromarray(segmentation_image)
    segmentation_image.show()


def evaluate_model(*, ann_train, train_ids):
    print("\n### Evaluation process ###")
    cocoEval = COCOeval(cocoGt=ann_train, cocoDt=None)  # iouType is "segm" by default
    cocoEval.params.imgIds = train_ids

    print("### Evaluates detections on every image and every category  ###\n")
    cocoEval.evaluate()
    print("### Accumulates the per-image, per-category evaluation ###\n")
    cocoEval.accumulate()
    print("### Display summary metrics of results ###\n")
    cocoEval.summarize()