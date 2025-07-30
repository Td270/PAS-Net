"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             cohen_kappa_score, roc_auc_score,
                             confusion_matrix)



def extra_metrics(logits: torch.Tensor, targets: torch.Tensor):

    probs  = logits.softmax(dim=1).cpu().numpy()     # (N, C)
    preds  = probs.argmax(1)
    y_true = targets.cpu().numpy()
    n_cls  = probs.shape[1]

    pre   = precision_score(y_true, preds, average='macro', zero_division=0)
    rec   = recall_score   (y_true, preds, average='macro', zero_division=0)
    f1    = f1_score       (y_true, preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, preds)


    if n_cls == 2:

        auc = roc_auc_score(y_true, probs[:, 1])
    else:
        auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')

    # --------- Specificity / Youden ----------
    cm = confusion_matrix(y_true, preds, labels=np.arange(n_cls))
    tn_list, fp_list = [], []
    for c in range(n_cls):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        tn_list.append(tn);  fp_list.append(fp)
    spec   = np.mean(np.array(tn_list) / (np.array(tn_list) + np.array(fp_list) + 1e-8))
    youden = rec + spec - 1

    return dict(Pre=pre, Recall=rec, Spec=spec,
                F1=f1, Kappa=kappa, Youden=youden, AUC=auc)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True
                    ):
    # TODO fix this for finetuning
    model.train(set_training_mode)
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        else:
            if targets.dtype != torch.long:  # one-hot/soft → hard label
                if targets.ndim > 1:
                    targets = targets.argmax(dim=1)
                targets = targets.long()
        # if not isinstance(criterion, torch.nn.modules.loss._Loss) or not isinstance(criterion, (torch.nn.KLDivLoss,)):
        #     if targets.dtype != torch.long and targets.ndim == 1:
        #         targets = targets.long()

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if isinstance(outputs, tuple):
                loss = 0.5 * criterion(outputs[0], targets) + 0.5 * criterion(outputs[1], targets)
            elif isinstance(outputs, list):
                loss_list = [criterion(o, targets) / len(outputs) for o in outputs]
                loss = sum(loss_list)
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if isinstance(outputs, list):
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
        else:
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

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            # Conformer
            if isinstance(output, list):
                loss_list = [criterion(o, target) / len(output)  for o in output]
                loss = sum(loss_list)
            # others
            else:
                loss = criterion(output, target)
        if isinstance(output, list):
            # Conformer
            logits_fused = output[0] + output[1]
            acc1_head1 = accuracy(output[0], target, topk=(1,))[0]
            acc1_head2 = accuracy(output[1], target, topk=(1,))[0]
            acc1_total = accuracy(output[0] + output[1], target, topk=(1,))[0]
            extra = extra_metrics(logits_fused, target)
            for k, v in extra.items():
                metric_logger.meters[k].update(v, n=images.size(0))
        else:
            # others
            acc1 = accuracy(output, target, topk=(1,))[0]

        batch_size = images.shape[0]
        if isinstance(output, list):
            metric_logger.update(loss=loss.item())
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
            metric_logger.meters['acc1'].update(acc1_total.item(), n=batch_size)
            metric_logger.meters['acc1_head1'].update(acc1_head1.item(), n=batch_size)
            metric_logger.meters['acc1_head2'].update(acc1_head2.item(), n=batch_size)
        else:
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if isinstance(output, list):
        print('* Acc@heads_top1 {heads_top1.global_avg:.3f} Acc@head_1 {head1_top1.global_avg:.3f} Acc@head_2 {head2_top1.global_avg:.3f} '
              'loss@total {losses.global_avg:.3f} loss@1 {loss_0.global_avg:.3f} loss@2 {loss_1.global_avg:.3f} '
              .format(heads_top1=metric_logger.acc1, head1_top1=metric_logger.acc1_head1, head2_top1=metric_logger.acc1_head2,
                      losses=metric_logger.loss, loss_0=metric_logger.loss_0, loss_1=metric_logger.loss_1))
        print('* Pre {Pre.global_avg:.3f} Re {Recall.global_avg:.3f} '
              'Spec {Spec.global_avg:.3f} F1 {F1.global_avg:.3f} '
              'κ {Kappa.global_avg:.3f} Youden {Youden.global_avg:.3f} '
              'AUC {AUC.global_avg:.3f}'
              .format(**metric_logger.meters))
    else:
        print('* Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
