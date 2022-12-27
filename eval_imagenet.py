import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
import time


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights):
    # Search Hyperparameters
    best_beta = torch.load('/home/hiyamanishi/CoOp/Tip-Adapter/caches/imagenet/best_beta_1shots.pt').item() 
    best_alpha = torch.load('/home/hiyamanishi/CoOp/Tip-Adapter/caches/imagenet/best_alpha_1shots.pt').item()


    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    print(affinity.shape, cache_values.shape, test_features.shape, cache_keys.shape)
    print(clip_logits.shape, cache_logits.shape)
    print(best_alpha, best_beta)
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))
    with open('tip_imagenetv2.txt', 'w') as f:
        f.write(str(cfg['shots']))
        f.write(' ')
        f.write(str(acc))


def run_tip_adapter_F(cfg, cache_keys, cache_values,test_features, test_labels, clip_weights, clip_model):
    
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    best_beta = torch.load('/home/hiyamanishi/CoOp/Tip-Adapter/caches/imagenet/best_beta_1shots.pt').item() 
    best_alpha = torch.load('/home/hiyamanishi/CoOp/Tip-Adapter/caches/imagenet/best_alpha_1shots.pt').item()
        # Eval
    adapter.eval()
    clip_logits = 100. * test_features @ clip_weights

    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-AdapterF's test accuracy: {:.2f}. ****\n".format(acc))
    with open('tipf_imagenetv2.txt', 'w') as f:
        f.write(str(cfg['shots']))
        f.write(' ')
        f.write(str(acc))



def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    test_loader = build_data_loader(data_source=dataset.test, batch_size=100, is_train=False, tfm=preprocess, shuffle=False)


    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys=torch.load('/home/hiyamanishi/CoOp/Tip-Adapter/caches/imagenet/best_F_1shots.pt')
    cache_values = torch.load('/home/hiyamanishi/CoOp/Tip-Adapter/caches/imagenet/cache_values_1shots.pt')
    # Pre-load val features
    #print("\nLoading visual features and labels from val set.")
    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    cache_keys=torch.load('/home/hiyamanishi/CoOp/Tip-Adapter/caches/imagenet/keys_1shots.pt')
    cache_values = torch.load('/home/hiyamanishi/CoOp/Tip-Adapter/caches/imagenet/cache_values_1shots.pt')
    run_tip_adapter(cfg, cache_keys, cache_values,test_features, test_labels, clip_weights)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    cache_keys=torch.load('/home/hiyamanishi/CoOp/Tip-Adapter/caches/imagenet/best_F_1shots.pt').T
    cache_values = torch.load('/home/hiyamanishi/CoOp/Tip-Adapter/caches/imagenet/cache_values_1shots.pt')
    run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model)
           

if __name__ == '__main__':
    main()