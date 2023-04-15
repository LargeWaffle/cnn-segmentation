import time

import torch
from torchvision import models
from torchvision.models.segmentation import FCN_ResNet101_Weights, DeepLabV3_ResNet101_Weights, \
    DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from imgutils import segment_map
from plotters import plot_results


def load_model(choice="dlab", train=False, feat_extract=True, nb_class=1):
    print()
    if choice == "dlab":
        print(f"Model is {choice}")
        w = DeepLabV3_ResNet101_Weights.DEFAULT
        m = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, weights=w)

    elif choice == "dlab_large":
        print(f"Model is {choice}")
        w = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        m = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True, weights=w)
    else:
        print(f"Model is FCN")
        w = FCN_ResNet101_Weights.DEFAULT
        m = models.segmentation.fcn_resnet101(pretrained=True, progress=True, weights=w)

    if feat_extract:
        for param in m.parameters():
            param.requires_grad = False

    if train:
        m = create_trainable_dlab(m, nb_class)

    params = [param for (name, param) in m.named_parameters() if param.requires_grad]

    return m, params


def create_trainable_dlab(model, nb_class):
    model.aux_classifier = None

    prev_channels = model.classifier[-1].in_channels
    model.classifier = DeepLabHead(prev_channels, nb_class)

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def inference(model, dataloader, colormap, device, nbinf=5):
    model = model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            print("Iteration %d" % i)
            inp = data.unsqueeze(0).to(device)

            st = time.time()
            out = model.to(device)(inp)['out']
            end = time.time()

            print(f"Inference took: {end - st:.2f}", )

            seg, overlay = segment_map(out, data, colormap)

            plot_results(data, seg, overlay)

            if i == nbinf:
                break
