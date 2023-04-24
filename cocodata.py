import os
from os import listdir
from os.path import isfile

import matplotlib.pyplot
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
import cv2


class CocoDataset(Dataset):
    def __init__(self, root, subset, transform=None, sup=False):
        print(f"\nLoading {subset} dataset")

        self.imgs_dir = os.path.join(root + "/images/", subset)

        ann_file = os.path.join(root + "/annotation/", f"instances_{subset}2017.json")
        self.coco = COCO(ann_file)

        self.sup = sup
        self.classes = self.coco.loadCats(self.coco.getCatIds())

        self.class_names = [cat['name'] for cat in self.classes]
        self.superclasses = list(set([cat['supercategory'] for cat in self.classes]))

        self.target_classes = self.superclasses if self.sup else self.classes

        self.target_classes_nb = len(self.target_classes) + 1

        self.img_ids = self.coco.getImgIds()

        self.transform = transform

    def assign_class(self, normal_class, attrname):
        for c in self.classes:
            if c['id'] == normal_class:
                return c[attrname]

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        anns = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        img_obj = self.coco.loadImgs(img_id)[0]

        img = Image.open(os.path.join(self.imgs_dir, img_obj['file_name'])).convert('RGB')

        mask = np.zeros(img.size[::-1], dtype=np.uint8)

        for ann in anns:
            class_name = self.assign_class(ann['category_id'], 'name')
            pixel_value = self.class_names.index(class_name) + 1
            mask = np.maximum(self.coco.annToMask(ann) * pixel_value, mask)

        if self.sup:
            for cl in self.classes:
                idx = mask == cl['id']
                class_index = self.assign_class(cl['id'], 'supercategory')
                mask[idx] = self.superclasses.index(class_index) + 1

            idx = mask >= self.target_classes_nb
            mask[idx] = 0

        mask = Image.fromarray(mask)

        if self.transform is not None:
            img = self.transform(img)
            img = T.ToTensor()(img)
            img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

            mask = self.transform(mask)
            mask = T.PILToTensor()(mask)

        return img, mask.long()

    def __len__(self):
        return len(self.img_ids)


class CocoTestDataset(Dataset):
    def __init__(self, root, subset, transform=None):
        print(f"\nLoading {subset} dataset")

        self.imgs_dir = os.path.join(root + "/images/", subset)
        self.img_names = [f for f in listdir(self.imgs_dir) if isfile(os.path.join(self.imgs_dir, f))]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.imgs_dir, self.img_names[idx])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.img_names)


def get_data(input_size, batch_size=64, sup=False):
    data_transforms = {
        'train': T.Compose([
            T.Resize(input_size, interpolation=F.InterpolationMode.BILINEAR),
            T.CenterCrop(input_size)
        ]),
        'val': T.Compose([
            T.Resize(input_size, interpolation=F.InterpolationMode.BILINEAR),
            T.CenterCrop(input_size),
        ]),
        'test': T.Compose([
            T.Resize(input_size, interpolation=F.InterpolationMode.BILINEAR),
            T.CenterCrop(input_size),
            T.ToTensor()
        ]),
    }

    coco_train = CocoDataset(root="data", subset="train", transform=data_transforms["train"], sup=sup)
    sub1 = torch.utils.data.Subset(coco_train, range(0, 10))

    train_dl = DataLoader(sub1, batch_size=batch_size, shuffle=True)

    coco_val = CocoDataset(root="data", subset="val", transform=data_transforms["val"], sup=sup)
    sub2 = torch.utils.data.Subset(coco_val, range(0, 5))

    val_dl = DataLoader(sub2, batch_size=batch_size, shuffle=True)

    coco_test = CocoTestDataset(root="data", subset="test", transform=data_transforms["test"])
    sub3 = torch.utils.data.Subset(coco_test, range(0, 10))

    test_dl = DataLoader(sub3, batch_size=None, shuffle=True)

    cats = ['unlabeled'] + coco_train.target_classes

    return train_dl, val_dl, test_dl, cats
