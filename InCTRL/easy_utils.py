import os
from os.path import isdir, isfile

import pandas as pd
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, accuracy_score, roc_curve

from open_clip.utils.checkpoint import load_checkpoint

categories_by_dataset = {
    'VisA': [
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
    'mvtec': ['bottle', 'cable', 'capsule', 'carpet', 'grid',
              'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
              'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
    'chexpert': [
        'chest',
    ],
    'riseholme': [
        'strawberry',
    ]
}

class AnomalyDataset(Dataset):
    def __init__(self, dataset_name, dataset_path, transform, split: str = 'test', category=None):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.transform = transform

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
                            if self.dataset_name == 'VisA':
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
        if grnd_truth_path != '':
            gnd_truth = Image.open(grnd_truth_path)
        else:
            try:
                gnd_truth = Image.fromarray(np.zeros_like(img, np.uint8), 'RGB')
            except ValueError:
                gnd_truth = img

        if self.transform:
            img = self.transform(img)
            gnd_truth = self.transform(gnd_truth)

        # wrapping the image into a list is needed since during testing
        # Normal images come from the few shots. Whereas during training
        # each task is constructed from 1 query image and multiple normal images
        return [img], cat, cls, anom_class, gnd_truth

    def __getitem__(self, index):
        if type(index) is slice:
            return [self._prepare_item(x) for x in range(*index.indices(len(self.data)))]
        return self._prepare_item(index)

    def __len__(self):
        return len(self.data)

    def recap(self):
        for k, v in self.by_category.items():
            print(f'Category {k} - Good {len(v[0])} - Defect {len(v[1])}')

    def get_df(self):
        good = []
        defect = []
        for (normal, defectious) in self.by_category.values():
            good.append(len(normal))
            defect.append(len(defectious))

        df = pd.DataFrame({'anomaly-free': good, 'anomalous': defect}, self.categories)
        return df


def construct_test_loader(dataset_name: str, dataset_path: str, transforms, category='candle', split='test'):
    dataset = AnomalyDataset(dataset_name=dataset_name,
                             dataset_path=dataset_path,
                             transform=transforms,
                             category=category,
                             split=split)

    dataset.recap()
    batch_size = 30

    return dataset, DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True, drop_last=False, shuffle=False)


def construct_test_loaders(dataset_name: str, dataset_path: str, transforms):
    categories = categories_by_dataset[dataset_name]

    loaders = []
    for c in categories:
        loaders.append(construct_test_loader(dataset_name, dataset_path, transforms, category=c))

    return loaders


def load_test_checkpoint(model, model_type: str, checkpoint_path: str):
    load_checkpoint(checkpoint_path, model, False)


@torch.no_grad()
def eval_model(model, loader, dataset, tokenizer, few_show_set):
    model.eval()

    total_label = torch.tensor([], dtype=torch.int32).cuda()
    total_pred = torch.Tensor([]).cuda()
    all_images = []

    for i, (images, categories, labels, class_defects, ground_truth) in enumerate(loader):
        labels = labels.cuda()

        preds, _ = model(tokenizer, images, categories, few_show_set[categories[0]])

        total_pred = torch.cat((total_pred, preds), 0)
        total_label = torch.cat((total_label, labels), 0)

        for image in images[0]:
            all_images.append(wandb.Image(image.permute(1, 2, 0).cpu().numpy()))

    total_pred = total_pred.cpu().numpy()
    total_label = total_label.cpu().numpy()

    preds = np.concatenate((total_pred.reshape((-1, 1)), total_label.reshape((-1, 1)), np.array(all_images).reshape((-1, 1))), axis=1)

    binary_pred = np.concatenate(((1 - total_pred).reshape((-1, 1)), total_pred.reshape((-1, 1))), axis=1)

    wandb.log({f'{dataset.category}_pred': wandb.Table(data=preds, columns=['total_pred', 'total_label', 'image'])})
    wandb.log({f'{dataset.category}_ROC': wandb.plot.roc_curve(total_label, binary_pred, labels=[f'{dataset.category}_normal', f'{dataset.category}_defect'])})
    wandb.log({f'{dataset.category}_PR': wandb.plot.pr_curve(total_label, binary_pred, labels=[f'{dataset.category}_normal', f'{dataset.category}_defect'])})

    #wandb.log({f'{dataset.category}_conf_mat': wandb.plot.confusion_matrix(preds=total_pred, y_true=total_label, class_names=[f'{dataset.category}_normal', f'{dataset.category}_defect'])})

    roc_auc, ap = aucPerformance(total_pred, total_label, prt=False)

    fpr, tpr, threshold = roc_curve(total_label, total_pred)
    precision, recall, pr_threshold = precision_recall_curve(total_label, total_pred)

    upper = 2 * (precision * recall)
    lower = precision + recall
    f1_scores = np.divide(upper, lower, out=np.zeros_like(upper), where=lower!=0.0)
    opt_pr_thresh = pr_threshold[np.argmax(f1_scores)]
    max_f1_score = np.max(f1_scores)

    opt_thresh = threshold[np.argmax(tpr - fpr)]
    concrete_pred = (total_pred > opt_thresh).astype(float)

    opt_acc = accuracy_score(total_label, concrete_pred)
    opt_precision = precision_score(total_label, concrete_pred)
    opt_recall = recall_score(total_label, concrete_pred)
    opt_f1_score = f1_score(total_label, concrete_pred)

    fpr_table = wandb.Table(data=[[x, y] for (x, y) in zip(threshold, fpr)], columns=['threshold', 'fpr'])
    fpr_line = wandb.plot.line(fpr_table, 'threshold', 'fpr', title=f'{dataset.category} False positive rate curve')

    tpr_table = wandb.Table(data=[[x, y] for (x, y) in zip(threshold, tpr)], columns=['threshold', 'tpr'])
    tpr_line = wandb.plot.line(tpr_table, 'threshold', 'tpr', title=f'{dataset.category} True positive rate curve')

    precision_table = wandb.Table(data=[[x, y] for (x, y) in zip(pr_threshold, precision)], columns=['threshold', 'precision'])
    precision_line = wandb.plot.line(precision_table, 'threshold', 'precision', title=f'{dataset.category} Precisison curve')

    recall_table = wandb.Table(data=[[x, y] for (x, y) in zip(pr_threshold, recall)], columns=['threshold', 'recall'])
    recall_line = wandb.plot.line(recall_table, 'threshold', 'recall', title=f'{dataset.category} Recall curve')

    wandb.log({
        f'{dataset.category}_fpr': fpr_line,
        f'{dataset.category}_tpr': tpr_line,
        f'{dataset.category}_precision': precision_line,
        f'{dataset.category}_recall': recall_line,
        f'{dataset.category}_ROC_AUC': roc_auc,
        f'{dataset.category}_AUC_PR': ap,
        f'{dataset.category}_opt_thresh': opt_thresh,
        f'{dataset.category}_opt_acc': opt_acc,
        f'{dataset.category}_opt_precision': opt_precision,
        f'{dataset.category}_opt_recall': opt_recall,
        f'{dataset.category}_opt_f1_score': opt_f1_score,
        f'{dataset.category}_opt_pr_thresh': opt_pr_thresh,
        f'{dataset.category}_max_f1_score': max_f1_score,
    })
    
    return roc_auc, ap, dataset.category, total_label, total_pred


def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap
