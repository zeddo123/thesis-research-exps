import argparse
import json
import os
import statistics

import numpy as np
import wandb
from tqdm import tqdm
import torch
from torchvision import transforms

import open_clip
from open_clip.model import get_cast_dtype
from engine_test import _convert_to_rgb
from easy_utils import construct_test_loader, construct_test_loaders, eval_model, load_test_checkpoint

parser = argparse.ArgumentParser('easy-inCTRL-runner')

parser.add_argument('-dn', '--dataset_name', type=str, default='VisA')
parser.add_argument('-dp', '--dataset_path', type=str)

parser.add_argument('-mt', '--model_type', type=str)
parser.add_argument('-mp', '--model_path', type=str)

parser.add_argument('-om', '--output_metrics', type=str, default='metrics.json')
parser.add_argument('-op', '--output_predictions', type=str, default='output.csv')

parser.add_argument('-fs', '--few_shot_path', type=str)

args = parser.parse_args()

device = torch.cuda.current_device()

transform = transforms.Compose([
    transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(240, 240)),
    _convert_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

cf = './open_clip/model_configs/ViT-B-16-plus-240.json'
with open(cf, 'r') as f:
    model_cfg = json.load(f)
embed_dim = model_cfg["embed_dim"]
vision_cfg = model_cfg["vision_cfg"]
text_cfg = model_cfg["text_cfg"]
cast_dtype = get_cast_dtype('fp32')
quick_gelu = False

model = open_clip.model.InCTRL(None, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)
model = model.cuda(device=device)

load_test_checkpoint(model, args.model_type, args.model_path)

tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

dataset, _ = construct_test_loader(args.dataset_name, args.dataset_path, transform, category=None)

few_shot_set = {}
shot = 0
for c in dataset.categories:
    few_shot_path = os.path.join(args.few_shot_path, c + ".pt")
    few_shot_set[c] = torch.load(few_shot_path)
    shot = len(few_shot_set[c])

#os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project='Anomaly Detection',
    config={
        "model": "InCTRL",
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "model_type": args.model_type,
        "model_path": args.model_path,
        "few_shot_path": args.few_shot_path,
        "run": "testing",
        "k-shot": shot,
        "device": device
    }
)

loaders = construct_test_loaders(args.dataset_name, args.dataset_path, transform)

roc_aucs = []
auc_prs = []
all_labels = []
all_preds = []

for dataset, loader in tqdm(loaders):
    roc_auc, auc_pr, cat, labels, preds = eval_model(model, loader, dataset, tokenizer, few_shot_set)
    roc_aucs.append(roc_auc)
    auc_prs.append(auc_pr)
    all_labels.append(labels)
    all_preds.append(preds)

wandb.log({'avg_roc_auc': statistics.mean(roc_aucs), 'avg_auc_pr': statistics.mean(auc_prs)})

all_preds = np.hstack(all_preds)
all_preds = np.concatenate(((1 - all_preds).reshape((-1, 1)), all_preds.reshape((-1, 1))), axis=1)
wandb.log({'ROC': wandb.plot.roc_curve(np.hstack(all_labels), all_preds, labels=['normal', 'defectious'])})
wandb.log({'ROC': wandb.plot.pr_curve(np.hstack(all_labels), all_preds, labels=['normal', 'defectious'])})

wandb.finish()
