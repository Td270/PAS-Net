from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, roc_auc_score
)
import numpy as np
import argparse
import time
import torch
from torchvision import transforms, datasets
from timm.models import create_model
from dataset.datasets import build_transform
from pathlib import Path
from torch.utils import data
from timm.utils import accuracy
import utils
from model import models
import matplotlib.pyplot as plt
# import cv2
import contextlib
import pandas as pd
import io
from fvcore.nn import FlopCountAnalysis, parameter_count
from torchinfo import summary

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_flops_and_params_fvcore(model, input_res):
    model.eval()
    dummy_input = torch.randn(1, 3, input_res, input_res).to(next(model.parameters()).device)
    try:
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total() / 1e9
    except Exception as e:
        print(f"FLOP analysis failed: {e}")
        total_flops = 0.0
    params = parameter_count(model)[''] / 1e6
    return round(params, 2), round(total_flops, 2)

def evaluate_one_weight(weight_path, args, test_loader, device):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
    ).to(device)

    # model = create_model('resnest50d', pretrained=False, num_classes=args.nb_classes).to(device)

    checkpoint = torch.load(weight_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])

        params_m, flops_g = get_flops_and_params_fvcore(model, args.input_size)
        print(f"Params: {params_m}M, FLOPs: {flops_g}G")


    model.eval()
    all_preds, all_labels, prob_list = [], [], []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            if isinstance(output, (list, tuple)):
                output = output[0] + output[1]
            else:
                output = output
            preds = torch.argmax(output, dim=1)
            probs = torch.softmax(output, dim=1)[:, 1]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            prob_list.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(prob_list)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred) * 100
    recall = recall_score(y_true, y_pred) * 100
    specificity = tn / (tn + fp) * 100
    f1 = f1_score(y_true, y_pred) * 100
    kappa = cohen_kappa_score(y_true, y_pred) * 100
    youden = recall + specificity - 100
    auc = roc_auc_score(y_true, y_prob) * 100

    return [acc, precision, recall, specificity, f1, kappa, youden, auc, tp, tn, fp, fn]


def main(args):
    print(args)
    device = torch.device(args.device)
    ROOT = Path("/xxx/data")
    test_image_path = ROOT / "test"

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    test_transform = build_transform(is_train=False, args=args)
    # pretrained_size = 224
    # pretrained_means = [0.485, 0.456, 0.406]
    # pretrained_stds = [0.2222, 5, 0.222, 5.2229]

    test_data = datasets.ImageFolder(
        root=test_image_path, transform=test_transform)

    test_loader = data.DataLoader(test_data, batch_size=args.batch_size, num_workers=4)

    # 'ConvNeXt_conformer_small_patch16'
    print(f"Creating model: {args.model}")
    results = []
    TP_sum, TN_sum, FP_sum, FN_sum = 0, 0, 0, 0
    for i in range(1, 4):
        weight_path = f"/xxx/HWMSA_conformer_small_patch16-320-0.0001-({i}).pth"
        print(f"Evaluating: {weight_path}")
        metrics = evaluate_one_weight(weight_path, args, test_loader, device)
        results.append(metrics[:8])
        TP_sum += metrics[8]
        TN_sum += metrics[9]
        FP_sum += metrics[10]
        FN_sum += metrics[11]

    columns = ["Acc", "Pre", "Recall", "Spec", "F1", "Kappa", "Youden", "AUC"]
    df = pd.DataFrame(results, columns=columns)
    summary = df.agg(['mean', 'std']).T
    summary['mean±std'] = summary.apply(lambda row: f"{row['mean']:.2f}±{row['std']:.2f}", axis=1)

    print("\n=== Summary (mean ± std across 10 runs) ===")
    print(summary['mean±std'])
    print(f"Label 0 (MDR): TN = {TN_sum}, FP = {FP_sum}")
    print(f"Label 1 (Sensitive): TP = {TP_sum}, FN = {FN_sum}")




if __name__ == '__main__':
    def get_args_parser():
        parser = argparse.ArgumentParser('HWMSA_conformer_small_patch16', add_help=False)
        parser.add_argument('--batch-size', default=8, type=int)
        parser.add_argument('--device', default='cuda:0')
        parser.add_argument('--input_size', type=int, default=320)
        parser.add_argument('--model', default='HWMSA_conformer_small_patch16', type=str, metavar='MODEL',
                            help='Name of model to train')
        parser.add_argument('--nb_classes', type=int, default=2)
        parser.add_argument('--drop', type=float, default=0.1, metavar='PCT',
                            help='Dropout rate (default: 0.)')
        parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                            help='Drop path rate (default: 0.1)')
        parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                            help='Drop block rate (default: None)')

        return parser

    parser = argparse.ArgumentParser('Conformer test', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
