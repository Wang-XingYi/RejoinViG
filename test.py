import argparse
import datetime
import os
import random

import torch.distributed as dist
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path
from torch.utils.data import DataLoader
from src.data_utils import TrainDataset,TestDataset
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from src.engine import *
from src.profiles import *
from src import samplers,utils,rejoinvig,losses


def get_args_parser():
    parser = argparse.ArgumentParser(
        'RejoinViG training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int) # 16 GPUs, so 2048 effective batch size

    # Model parameters
    parser.add_argument('--model', default='RejoinViG_S', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--img_w', type=int, default=224)
    parser.add_argument('--img_h', type=int, default=224)
    parser.add_argument('--dataset_dir', type=str, default='Dataset')
    parser.add_argument('--dataset_txt', type=str, default='Test_full.txt')
    parser.add_argument('--save_txt', type=str, default="./logs/01_synthesis_dataset_record_log.txt")
    parser.add_argument('--pred_file', type=str, default="./logs/01_synthesis_pred_file.pkl")
    parser.add_argument('--nb-classes', type=int, default=5, help='the number of classes')


    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true',default=True,
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)

    parser.set_defaults(sync_bn=True)


    return parser
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    os.environ['PYTHONHASHSEED']=str(seed)
    # cudnn.benchmark = True

def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    set_seed(seed)


    # read data
    exp_name = './'
    val_data = TestDataset(data_path=exp_name,dataset_dir=args.dataset_dir,dataset_txt=args.dataset_txt,
                            WIDTH=args.img_w, HEIGHT=args.img_h)
    data_loader_val = DataLoader(dataset=val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)


    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        distillation=False,
        pretrained=args.eval,
        fuse=args.eval,
    )

    model.to(device)


    tmp = f"{args.output_dir}/model_best.pth"
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
    test_stats = test(data_loader_val, model, device, args)
    print(f"Accuracy of the network on the {len(val_data)} test images: {test_stats['acc1']:.1f}%")


if __name__ == '__main__':
    # torch.cuda.set_device(1)

    parser = argparse.ArgumentParser(
        'RejoinViG training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
