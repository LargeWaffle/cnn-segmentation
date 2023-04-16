import os
import numpy as np

import cv2
from PIL import Image
import torch
import torchvision.transforms.functional as F
from pycocotools.coco import COCO
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


def get_class_name(class_id, cats):
    for cat in cats:
        if cat['id'] == class_id:
            return cat['name']
    return "None"


class CocoDataset(Dataset):
    def __init__(self, root, subset, transform=None):
        print(f"\nLoading {subset} dataset")
        dataset_path = os.path.join(root + "/images/", subset)

        ann_file = os.path.join(root + "/annotation_folder/annotations/", f"instances_{subset}2017.json")

        self.imgs_dir = dataset_path

        self.coco = COCO(ann_file)

        self.classes = self.coco.loadCats(self.coco.getCatIds())

        self.class_names = [cat['name'] for cat in self.classes]
        self.superclasses = set([cat['supercategory'] for cat in self.classes])

        self.img_ids = self.coco.getImgIds()

        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_obj = self.coco.loadImgs(img_id)[0]
        anns = self.coco.loadAnns(self.coco.getAnnIds(img_id))

        img = Image.open(os.path.join(self.imgs_dir, img_obj['file_name'])).convert('RGB')

        mask = np.zeros(img.size, dtype=np.uint8)

        for ann in anns:
            class_name = get_class_name(ann['category_id'], self.classes)
            pixel_value = self.class_names.index(class_name) + 1
            new_mask = cv2.resize(self.coco.annToMask(ann) * pixel_value, img.size[::-1])
            mask = np.maximum(new_mask, mask)

        mask = Image.fromarray(mask)
        # .convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            mask = self.transform(mask)

        return img, mask

    def __len__(self):
        return len(self.img_ids)


def get_data(input_size, batch_size=64):
    data_transforms = {
        'train': T.Compose([
            # T.RandomResizedCrop(input_size),
            # T.RandomHorizontalFlip(),
            T.Resize(input_size, interpolation=F.InterpolationMode.BILINEAR),
            T.CenterCrop(input_size),
            T.ToTensor(),
        ]),
        'val': T.Compose([
            T.Resize(input_size, interpolation=F.InterpolationMode.BILINEAR),
            T.CenterCrop(input_size),
            T.ToTensor(),
        ]),
        'test': T.Compose([
            T.Resize(input_size, interpolation=F.InterpolationMode.BILINEAR),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    coco_train = CocoDataset(root="../data", subset="train", transform=data_transforms["train"])
    sub1 = torch.utils.data.Subset(coco_train, range(0, 10))

    train_dl = DataLoader(sub1, batch_size=batch_size, shuffle=True)

    coco_val = CocoDataset(root="../data", subset="val", transform=data_transforms["val"])
    sub2 = torch.utils.data.Subset(coco_val, range(0, 5))

    val_dl = DataLoader(sub2, batch_size=batch_size, shuffle=True)

    test_imgs = ImageFolder(root="../data/images/folder/", transform=data_transforms["test"])
    test_dl = DataLoader(test_imgs, batch_size=None, shuffle=True)

    return train_dl, val_dl, test_dl, len(coco_train.class_names)
