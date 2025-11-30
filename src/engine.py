"""
Train and eval functions used in main.py
"""
import math
import pickle
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from tqdm import tqdm

from src.BCEFocalLoss import FocalLoss
from .losses import DistillationLoss
from . import utils as utils
import torch.nn.functional as F

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, batch_size: int, gradient_accumulation: bool, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    sample_num = 0
    train_bar = tqdm(data_loader)  # 进度条
    running_results = {'batch_sizes': 0, 'loss': 0, 'acc': 0}

    if gradient_accumulation == True:
        batch_idx = 0
        accum_iter = 4 # making lambda 512 (128 * 4) batch size = 2048 (128 * 16) effective batch size

        # for inputs, labels in metric_logger.log_every(
        #         data_loader, print_freq, header, batch_idx):
        for i, batch_value in enumerate(train_bar):
            inputs = batch_value[0].float()
            labels = batch_value[1]

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sample_num += labels.shape[0]

            if mixup_fn is not None:
                inputs, labels = mixup_fn(inputs, labels)

            with torch.cuda.amp.autocast():     # if True:
                # preds = model(inputs)
                preds = model(inputs)
                # loss = criterion(inputs, preds, labels)
                loss = criterion(preds, labels)

            loss_value = loss.item() # gradient stored, until we call optimizer.zero_grad()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader)):
                optimizer.zero_grad()

                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(
                    optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                            parameters=model.parameters(), create_graph=is_second_order)

                torch.cuda.synchronize()
                if model_ema is not None:
                    model_ema.update(model)

                metric_logger.update(loss=loss_value)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            batch_idx += 1
    else:
        # for inputs, labels in metric_logger.log_every(
        #         data_loader, print_freq, header):
        for i, batch_value in enumerate(train_bar):
            inputs = batch_value[0].float()
            labels = batch_value[1]
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sample_num += labels.shape[0]

            # 混合增强方法，旨在通过混合输入样本和标签来提高模型的泛化能力
            if mixup_fn is not None:
                inputs, labels = mixup_fn(inputs, labels)

            with torch.cuda.amp.autocast():
                preds = model(inputs)
                # loss = criterion(inputs, preds, labels)
                loss = criterion(preds, labels)

            loss_value = loss.item() # gradient stored, until we call optimizer.zero_grad()

            pred_classes = torch.max(preds, dim=1)[1]
            accu_num = torch.eq(pred_classes, labels).sum()
            running_results['acc'] += accu_num.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()

            running_results['loss'] += loss.item()
            train_bar.set_description(desc='[%d/%d] Loss: %.4f acc: %.4f' % (
                epoch, 300, running_results['loss'] / (i+1),
                running_results['acc'] / sample_num
            ))

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                        parameters=model.parameters(), create_graph=is_second_order)

            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    batch_idx = 0

    # switch to evaluation mode
    model.eval()
    val_bar = tqdm(data_loader)  # 验证集的进度条


    # for images, target in metric_logger.log_every(data_loader, 10, header):
    for i, batch_value in enumerate(val_bar):
        images = batch_value[0].float()
        target = batch_value[1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        # compute output
        with torch.cuda.amp.autocast():
            # output = model(images)
            # loss = criterion(output, target)
            output = model(images)
            loss = criterion(output, target)

        acc1, acc3 = accuracy(output, target, topk=(1, 3))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@3 {top3.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top3=metric_logger.acc3, losses=metric_logger.loss))
    print(output.mean().item(), output.std().item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test(data_loader, model, device,args):
    criterion = torch.nn.CrossEntropyLoss()
    criterion=FocalLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    batch_idx = 0
    running_results = {'batch_sizes': 0, 'loss': 0, 'acc': 0}
    sample_num = 0

    # switch to evaluation mode
    model.eval()
    val_bar = tqdm(data_loader)  # 验证集的进度条
    f = open(args.save_txt, "w", encoding='utf-8')
    f.write(f'source_img target_img pre_probability pre_classes label\n')
    pred_file = open(args.pred_file, 'wb')


    # for images, target in metric_logger.log_every(data_loader, 10, header):
    for i, batch_value in enumerate(val_bar):
        images = batch_value[0].float()
        target = batch_value[1]
        img_names = batch_value[2]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sample_num += target.shape[0]


        # compute output
        with torch.cuda.amp.autocast():
            # output = model(images)
            # loss = criterion(output, target)
            output = model(images)
            loss = criterion(output, target)

        running_results['loss'] += loss.item()

        # 将 logits 转换为概率
        probabilities = F.softmax(output, dim=1)
        # 获取最大概率值
        max_probability, max_index = torch.max(probabilities, dim=1)
        for j in range(images.size(0)):
            f.write(f'{img_names[0][j]} {img_names[1][j]} {max_probability[j].item()} {max_index[j].item()} {target[j]}\n')
            pickle.dump(output[j].cpu(), pred_file)

        acc1, acc3 = accuracy(output, target, topk=(1, 3))

        pred_classes = torch.max(output, dim=1)[1]
        accu_num = torch.eq(pred_classes, target).sum()
        running_results['acc'] += accu_num.item()

        val_bar.set_description(desc='[%d/%d] Loss: %.4f acc: %.4f' % (
            1,1, running_results['loss'] / (i + 1),
            running_results['acc'] / sample_num
        ))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@3 {top3.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top3=metric_logger.acc3, losses=metric_logger.loss))
    print(output.mean().item(), output.std().item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
