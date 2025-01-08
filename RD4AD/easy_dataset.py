import os
from os.path import isdir, isfile

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

categories_by_dataset = {
    'visa': [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ],
    'mvtec': [
        'carpet',
        'bottle',
        'hazelnut',
        'leather',
        'cable',
        'capsule',
        'grid',
        'pill',
        'transistor',
        'metal_nut',
        'screw',
        'toothbrush',
        'zipper',
        'tile',
        'wood'
    ],
    'riseholme': ['strawberry'],
    'chexpert': ['chest'],
}

class AnomalyDataset(Dataset):
    def __init__(self, dataset_name, dataset_path, transform, gt_transform, split: str = 'test', category=None):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.transform = transform
        self.gt_transform = gt_transform

        self.categories = []
        self.by_category = {}
        self.category_gt = {}
        self.data = []

        self.category = category

        if category is None:
            for f in os.listdir(dataset_path):
                if isdir(os.path.join(dataset_path, f)):
                    self.categories.append(f)
                    self.by_category[f] = ([], [])
                    self._load_categorie(f, split)
        else:
            # In case a single category is selected
            self.categories.append(category)
            self.by_category[category] = ([], [])
            self._load_categorie(category, split)

    def _load_categorie(self, category, split):
        gnd_truth = f'{self.dataset_path}/{category}/ground_truth'
        test = f'{self.dataset_path}/{category}/test'
        train = f'{self.dataset_path}/{category}/train'

        loc = test if split == 'test' else train

        # Load ground truth
        if self.category_gt.get(category) is None:
            self.category_gt[category] = []

        for root, _, files in os.walk(gnd_truth):
            for file in files:
                self.category_gt[category].append(f'{root}/{file}')

        for f in os.listdir(loc):
            if isdir(os.path.join(loc, f)):
                for entry in os.listdir(f'{loc}/{f}'):
                    if isfile(os.path.join(loc, f, entry)):
                        image = f'{loc}/{f}/{entry}'

                        c = 0 if f == 'good' else 1
                        if c == 1:
                            file_name, _ = os.path.splitext(entry)
                            if self.dataset_name in ('visa', 'VisA'):
                                gnd_file_name = f'{file_name}.png'
                            else:
                                gnd_file_name = f'{file_name}_mask.png'

                            gnd = f'{gnd_truth}/{f}/{gnd_file_name}'

                            if self.dataset_name in ('chexpert', 'riseholme'):
                                gnd = ''
                        else:
                            gnd = ''

                        self.by_category[category][c].append((image, category, c, f, gnd))
                        self.data.append((image, category, c, f, gnd))

    def _prepare_item(self, index):
        if index >= len(self.data):
            return None
        img_path, cat, cls, anom_class, grnd_truth_path = self.data[index]

        img = Image.open(img_path)
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)

        if grnd_truth_path != '':
            gnd_truth = Image.open(grnd_truth_path).convert('1')
            if self.gt_transform:
                gnd_truth = self.gt_transform(gnd_truth)
        else:
            gnd_truth = torch.zeros([1, img.size()[-2], img.size()[-2]])


        return img, cat, cls, anom_class, gnd_truth

    def __getitem__(self, index):
        if type(index) is slice:
            return [self._prepare_item(x) for x in range(*index.indices(len(self.data)))]
        return self._prepare_item(index)

    def __len__(self):
        return len(self.data)

    def recap(self):
        for k, v in self.by_category.items():
            print(f'Category {k} - Good {len(v[0])} - Defect {len(v[1])}')

