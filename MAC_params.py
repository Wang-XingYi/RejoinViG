import argparse
import datetime
import os
import torch.distributed as dist
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path
from torch.utils.data import DataLoader
from data_utils import TrainDataset,TestDataset
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import rejoinvig
from engine import *
import losses
from profiles import *
import samplers
import utils
from torchprofile import profile_macs

def get_args_parser():
    parser = argparse.ArgumentParser(
        'GreedyViG training and evaluation script', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='GreedyViG_S', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', default=False, type=str)
    parser.add_argument('--num-classes', default=5, type=int)

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    model = create_model(
        args.model,
        num_classes=args.num_classes,
        distillation=False, # 是否知识蒸馏
        pretrained=args.eval,
        fuse=args.eval,
    )

    tmp = f"outputs/model_best.pth"
    if os.path.exists(tmp):
        args.resume = tmp
    flag = os.path.exists(args.resume)
    if args.resume and flag:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    input = torch.randn([1, 6, 224, 224])

    model.eval()
    macs = profile_macs(model, input)
    model.train()
    params=sum([m.numel() for m in model.parameters()])
    macs_in_g=macs/1e9
    params_in_m=params/1e6
    print(f'model MACs:{macs_in_g:.1f}G, param count:{params_in_m:.1f}, input_size:{input.shape}')




if __name__ == '__main__':
    # torch.cuda.set_device(1)

    parser = argparse.ArgumentParser(
        'GreedyViG training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
