import wandb

import argparse

from generate_few_shots import generate_fewshots
from easy_test import test




def benchmark(config):
    # generate few shot folder
    path = generate_fewshots(config.dataset_name, config.dataset_path, config.fewshots)

    auc, pr = test(config.dataset_name, config.dataset_path, config.model_type, config.model_path, path)

    
    wandb.log({'avg_auc_roc': auc})
    return auc



if __name__ == '__main__':
    parser = argparse.ArgumentParser('benchmark')

    parser.add_argument('-dn', '--dataset_name', type=str, default='VisA')
    parser.add_argument('-dp', '--dataset_path', type=str)

    parser.add_argument('-mt', '--model_type', type=str)
    parser.add_argument('-mp', '--model_path', type=str)
    parser.add_argument('-fs', '--fewshots', type=int)

    args = parser.parse_args()

    benchmark(args)
    
