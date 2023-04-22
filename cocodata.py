import os
from os import listdir
from os.path import isfile

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocostuffhelper import cocoSegmentationToSegmentationMap
from torch.utils.data import DataLoader, Dataset


class CocoDataset(Dataset):
    def __init__(self, root, subset, transform=None):
        print(f"\nLoading {subset} dataset")

        self.imgs_dir = os.path.join(root + "/images/", subset)

        ann_file = os.path.join(root + "/annotation_folder/stuff_annotations/", f"stuff_{subset}2017.json")
        self.coco = COCO(ann_file)

        self.classes = self.coco.loadCats(self.coco.getCatIds())

        self.class_names = sorted([cat['name'] for cat in self.classes])
        self.superclasses = sorted(set([cat['supercategory'] for cat in self.classes]))

        self.img_ids = self.coco.getImgIds()

        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_obj = self.coco.loadImgs(img_id)[0]

        img = Image.open(os.path.join(self.imgs_dir, img_obj['file_name'])).convert('RGB')

        mask = cocoSegmentationToSegmentationMap(self.coco, img_id)
        mask = Image.fromarray(mask)
        # .convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            mask = self.transform(mask)

        return img, mask

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
            T.ToTensor()
        ]),
    }

    coco_train = CocoDataset(root="../data", subset="train", transform=data_transforms["train"])
    sub1 = torch.utils.data.Subset(coco_train, range(0, 20))

    train_dl = DataLoader(sub1, batch_size=batch_size, shuffle=True)

    coco_val = CocoDataset(root="../data", subset="val", transform=data_transforms["val"])
    sub2 = torch.utils.data.Subset(coco_val, range(0, 10))

    val_dl = DataLoader(sub2, batch_size=batch_size, shuffle=True)

    coco_test = CocoTestDataset(root="../data", subset="test", transform=data_transforms["test"])
    sub3 = torch.utils.data.Subset(coco_test, range(0, 50))

    test_dl = DataLoader(sub3, batch_size=None, shuffle=True)

    return train_dl, val_dl, test_dl
