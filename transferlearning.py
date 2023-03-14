import tensorflow_hub as hub


def get_model():
    print("### Model from Tensor Hub ###")
    model = hub.load('https://tfhub.dev/google/HRNet/coco-hrnetv2-w48/1')

    return model
