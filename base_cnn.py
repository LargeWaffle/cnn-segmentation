import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


class CNN:

    def __init__(self, in_shape, nb_class):
        # Initialize the model
        model = Sequential()

        # convolutional layers with pooling and dropout
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=in_shape, padding="same"))
        model.add(MaxPooling2D())

        model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
        model.add(MaxPooling2D())

        model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
        model.add(MaxPooling2D())

        model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
        model.add(MaxPooling2D())
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=nb_class)])

        self.model = model
