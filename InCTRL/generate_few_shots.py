import os
import argparse

import torch
from torchvision import transforms

from engine_test import _convert_to_rgb
from easy_utils import construct_test_loader, categories_by_dataset


def generate_fewshots(dataset_name, dataset_path, kshots):
    transform = transforms.Compose([
        transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(240, 240)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

    for cls in categories_by_dataset[dataset_name]:
        dataset, _ = construct_test_loader(dataset_name, dataset_path, transform, category=cls, split='train')
        supporting_set = dataset[:kshots]
        supporting_set = [k[0][0] for k in supporting_set]
        print(len(supporting_set))

        if not os.path.exists(f'{dataset_name}_fewshot_{kshots}'):
            os.makedirs(f'{dataset_name}_fewshot_{kshots}')

        torch.save(supporting_set, f'{dataset_name}_fewshot_{kshots}/{cls}.pt')

    return f'{dataset_name}_fewshot_{kshots}/'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate-few_shots')

    parser.add_argument('-dn', '--dataset_name', type=str, default='VisA')
    parser.add_argument('-dp', '--dataset_path', type=str)
    parser.add_argument('-k', '--kshots', type=int, default=10)

    args = parser.parse_args()

    generate_fewshots(args.dataset_name, args.dataset_path, args.kshots)

