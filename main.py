import base_cnn as cnn

nb_class = 2

cnn_model = cnn.CNN(in_shape=x_train.shape[1:], nb_class=nb_class)
model = cnn_model.model

# Fit the model to the MNIST data
history = model.fit(x_train, y_train, batch_size=128, epochs=15, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy {test_acc}")
