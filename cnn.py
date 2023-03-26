from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, \
    BatchNormalization, Concatenate, Activation, Add
from keras.losses import CategoricalCrossentropy
from keras.metrics import Precision, Recall, AUC
from keras.models import Model
from keras.optimizers import Adam
from pycocotools.cocoeval import COCOeval
from tensorflow import saved_model

# import tensorflow_hub as hub


def get_premade_model():
    print("### Model from TensorFlow Hub ###")
    print("### HRNet_coco-hrnetv2-w48_1 ###")
    # model = hub.load('https://tfhub.dev/google/HRNet/coco-hrnetv2-w48/1')
    model = saved_model.load("models/HRNet/")
    print("### Model loaded ###")

    return model


def resnet_block(inputs, activation_function="relu", conv_filters=64):
    x = Conv2D(conv_filters, 3, activation="relu", padding="same", strides=1, kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation_function)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(conv_filters, 3, activation="relu", strides=1, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation(activation_function)(x)
    x = Dropout(0.25)(x)

    # format input to have compatible shape, it may degrade the training
    inp = Conv2D(kernel_size=1, strides=1, filters=conv_filters, padding="same")(inputs)

    resnet = Add()([inp, x])

    return Model(inputs, resnet)


def upsampling_layer(inputs, residual, conv_filters=64):
    x = UpSampling2D(size=(2, 2))(inputs)
    x = Conv2D(conv_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(x)

    x = Concatenate(axis=3)([residual, x])
    x = Conv2D(conv_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    upsample = Conv2D(conv_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

    return upsample


def get_model(input_size, num_classes, before_pooling=1, nb_pooling=4):
    print("### Personnal model ###")
    conv_filter_size = [64, 128, 256, 512]
    encoder_blocs = []

    in_x = Input(input_size, name="original_input")
    x = in_x

    # encoder part
    for i in range(nb_pooling):
        for _ in range(before_pooling):
            submodel = resnet_block(x, conv_filters=conv_filter_size[i])
            x = submodel(x)
            encoder_blocs.append(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)

    # end of the encoder part of the model

    x = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Dropout(0.5)(x)

    conv_filter_size.reverse()
    encoder_blocs.reverse()

    # decoder part
    for conv_filters, enc_block in zip(conv_filter_size, encoder_blocs):
        x = upsampling_layer(x, enc_block, conv_filters)

    x_out = Conv2D(num_classes, 3, activation='softmax', padding='same', kernel_initializer='he_normal')(x)

    model = Model(inputs=in_x, outputs=x_out)

    opt = Adam(learning_rate=1e-4)
    loss_function = CategoricalCrossentropy()

    print("### Compiling model ###")
    model.compile(optimizer=opt, loss=loss_function, metrics=['accuracy', Precision(), Recall(), AUC()])

    return model


def train_model(*, model, train_data, val_data, steps, val_steps, epochs):
    print("### Model training ###")

    # Start the training process
    history = model.fit(x=train_data,
                        validation_data=val_data,
                        steps_per_epoch=steps,
                        validation_steps=val_steps,
                        epochs=epochs,
                        verbose=True)

    return history


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
