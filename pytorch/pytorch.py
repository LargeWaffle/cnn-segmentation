import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as dset
# Apply the transformations needed
import torchvision.transforms as T
from torchvision import models
from torchvision.models.segmentation import FCN_ResNet101_Weights, DeepLabV3_ResNet101_Weights

img_size = 640

fcn = models.segmentation.fcn_resnet101(weights=FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).eval()
dlab = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).eval()

"""
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)

# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)

colormap = []
for hexcode in mcolors.CSS4_COLORS.values():
    r, g, b = [int(round(x * 255)) for x in mcolors.to_rgb(hexcode)]
    colormap.append((r, g, b))

label_colors = np.array([(0, 0, 0),  # 0=background
                         # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                         (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                         # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                         (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                         # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                         (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                         # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                         (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])


# Define the helper function
def decode_segmap(image, nc=21):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = colormap[l][0]
        g[idx] = colormap[l][1]
        b[idx] = colormap[l][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


transformations = T.Compose([
    T.Resize(img_size),
    T.CenterCrop(img_size),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def segment(net, img, dev):
    inp = transformations(img).unsqueeze(0).to(dev)

    out = net.to(dev)(inp)['out']

    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

    rgb = decode_segmap(om)

    rs = T.Compose([
        T.ToPILImage(),
        T.Resize(size=(img.size[::-1]))
    ])

    result = rs(rgb)

    f = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.axis('off')
    plt.show()


test_imgs = dset.ImageFolder(root="data/images/folder/")

dataloader = torch.utils.data.DataLoader(test_imgs, batch_size=None, shuffle=True)

for i, (data, _) in enumerate(dataloader):
    print("Iteration %d" % i)
    with torch.no_grad():

        st = time.time()
        segment(dlab, data, device)
        end = time.time()

        print(f"Inference took: {end - st:.2f}", )

        if i == 5:
            break
