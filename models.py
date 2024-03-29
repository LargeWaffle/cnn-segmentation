import time

import torch
import torchvision.transforms as T
from torchvision import models
from torchvision.models.segmentation import FCN_ResNet101_Weights, DeepLabV3_ResNet101_Weights, \
    DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from imgutils import segment_map
from tools import plot_results


def load_model(choice="dlab", train=False, feat_extract=False, nb_class=1):
    print()

    if choice == "dlab":
        print(f"Model is {choice}")
        w = DeepLabV3_ResNet101_Weights.DEFAULT
        m = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, weights=w)

    elif choice == "dlab_large":
        print(f"Model is {choice}")
        w = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        m = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True, weights=w)
    elif choice == "fcn":
        print(f"Model is FCN")
        w = FCN_ResNet101_Weights.DEFAULT
        m = models.segmentation.fcn_resnet101(pretrained=True, progress=True, weights=w)
    else:
        return

    m.aux_classifier = None

    if train:
        m = create_trainable_dlab(m, nb_class)

    if feat_extract:
        for param in m.parameters():
            param.requires_grad = False

        for param in m.classifier.parameters():
            param.requires_grad = True

    params = [param for (name, param) in m.named_parameters() if param.requires_grad]

    return m, params


def create_trainable_dlab(model, nb_class):

    sample_input = torch.randn(1, 3, 32, 32)  # batch size 1, RGB input image of size 520x520
    backbone_output = model.backbone(sample_input)['out']  # get output of backbone module
    prev_channels = backbone_output.shape[1]
    model.classifier = DeepLabHead(prev_channels, nb_class)

    return model


def inference(model, dataloader, cats, nb_class, device, nbinf=5):
    model = model.eval()

    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(nb_class)])[:, None] * palette
    colormap = (colors % 255).numpy().astype("uint8")

    with torch.no_grad():
        for i, img in enumerate(dataloader):
            print("Iteration %d" % i)
            inp = img.unsqueeze(0).to(device)
            inp = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(inp)

            st = time.time()
            out = model.to(device)(inp)['out']
            end = time.time()

            print(f"Inference took: {end - st:.2f}", )

            f_img = img.permute(1, 2, 0)
            seg, overlay, cnames = segment_map(out, f_img, colormap, cats, nb_class)

            plot_results(f_img, seg, overlay, cnames)

            if i == nbinf:
                break
