import os
import argparse
import random

from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import argparse
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

from dataset import MVTecDataset
from dataset import get_data_transforms
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from main import loss_function, setup_seed, count_parameters, loss_concat
from test import evaluation, visualization, test

from easy_dataset import AnomalyDataset, categories_by_dataset
from easy_metrics import wandb_metric_logs
import wandb


parser = argparse.ArgumentParser('easy-rd4ad-trainer')
parser.add_argument('-e', '--epochs', type=int, default=200)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.005)
parser.add_argument('-bs', '--batch_size', type=int, default=16)
parser.add_argument('-ims', '--image_size', type=int, default=256)

parser.add_argument('-dn', '--dataset_name', type=str)
parser.add_argument('-dp', '--dataset_path', type=str)

parser.add_argument('-om', '--output_metrics', type=str)
parser.add_argument('-op', '--output_predictions', type=str)

args = parser.parse_args()

device = torch.cuda.current_device()

step = 0

def test(args, _class_):
    global step
    data_transform, gt_transform = get_data_transforms(args.image_size, args.image_size)
    ckp_path = f'./checkpoints/wres50_{_class_}.pth'
    train_data = AnomalyDataset(args.dataset_name, args.dataset_path, data_transform, gt_transform, split='train', category=_class_)
    test_data = AnomalyDataset(args.dataset_name, args.dataset_path, data_transform, gt_transform, split='test', category=_class_)

    #train_data = train_data[:20]
    train_data = random.choices(train_data, k=20)
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    #test_path = f'{args.dataset_path}/{_class_}'
    #test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder.to(device)

    wandb.watch(decoder, log='all')

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=args.learning_rate, betas=(0.5,0.999))

    for epoch in tqdm(range(args.epochs), desc=f'{_class_} epochs', unit='epoch'):
        bn.train()
        decoder.train()
        loss_list = []
        for imgs, _, labels, class_defect, gndtruth in tqdm(train_loader, desc=f'{_class_} Training'):
            img = imgs.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))#bn(inputs))
            loss = loss_function(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        #print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, args.epochs, np.mean(loss_list)))
        wandb.log({f'{_class_}_loss': np.mean(loss_list)}, step=step)

        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px, table, roc, pr, _, _, _ = evaluation(encoder, bn, decoder, test_loader, device, _class_=_class_)

            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            wandb.log({f'{_class_}_pixel_auroc': auroc_px, f'{_class_}_sample_auroc': auroc_sp, f'{_class_}pixel_aupro': aupro_px}, step=step)

            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
        step += 1

    #wandb.log({f'{_class_}_pred': table, f'{_class_}_ROC': roc, f'{_class_}_PR': pr, f'{_class_}_ROC_AUC': auroc_sp})

    auroc_px, auroc_sp, aupro_px, table, roc, pr, y_true, y_pred, y_image = evaluation(
            encoder, bn, decoder, test_loader, device, _class_=_class_)
    wandb_metric_logs(_class_, np.array(y_image), np.array(y_pred), np.array(y_true))

    return auroc_px, auroc_sp, aupro_px

wandb.init(
    project='Anomaly Detection',
    config={
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "run": "training",
        "model": "RD4AD",
        "device": device
    }
)

for c in categories_by_dataset[args.dataset_name]:
    print(f'Training T-S model on {c}')
    test(args, c)
