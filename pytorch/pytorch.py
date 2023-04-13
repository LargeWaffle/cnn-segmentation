import time

import torch.utils.data
import torchvision.datasets as dset

from imgutils import segment_map

from torchvision import models
from torchvision.models.segmentation import FCN_ResNet101_Weights, DeepLabV3_ResNet101_Weights, \
    DeepLabV3_MobileNet_V3_Large_Weights


def load_model(choice="dlab"):
    if choice == "dlab":
        w = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        m = models.segmentation.deeplabv3_resnet101(weights=w)

    elif choice == "dlab_large":
        w = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        m = models.segmentation.deeplabv3_mobilenet_v3_large(weights=w)
    else:
        w = FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        m = models.segmentation.fcn_resnet101(weights=w)

    return m.eval(), w


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Segmentation project running on", device)

test_imgs = dset.ImageFolder(root="../data/images/folder/")

dataloader = torch.utils.data.DataLoader(test_imgs, batch_size=None, shuffle=True)

model, w = load_model()
transformations = w.transforms()

for i, (data, _) in enumerate(dataloader):
    print("Iteration %d" % i)
    inp = transformations(data).unsqueeze(0).to(device)

    st = time.time()
    with torch.no_grad():
        out = model.to(device)(inp)['out']
    end = time.time()

    print(f"Inference took: {end - st:.2f}", )

    segment_map(out, data)

    if i == 5:
        break
