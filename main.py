import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torchvision import transforms, datasets
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import datetime
from dataset.my_dataset import MyDataSet
import argparse
import datetime
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils import data
from torchvision import transforms, datasets
import json
import copy
from pathlib import Path
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from dataset.datasets import build_dataset,build_transform
from engine import train_one_epoch, evaluate
from dataset.samplers import RASampler
import utils
from thop import profile
from model import models



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args_parser():
    parser = argparse.ArgumentParser('HWMSA_conformer_small_patch16', add_help=False)
    parser.add_argument('--batch-size', default=42, type=int)
    parser.add_argument('--epochs', default=150, type=int)

    # Model parameters
    # Conformer_small_patch16
    # convnext_tiny
    # deit_tiny_patch16_224
    # deit_small_patch16_224
    # deit_base_patch16_224
    parser.add_argument('--model', default='HWMSA_conformer_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=320, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    #
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='/xxx/data', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', type=int, default=2)
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'CIFAR10', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    parser.add_argument('--evaluate-freq', type=int, default=1, help='frequency of perform evaluation (default: 5)')
    parser.add_argument('--output_dir', default='./model_weights',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=6050, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='', help='url used to set up distributed training')

    # ceit leff module
    parser.add_argument('--leff_local_size', default=3, type=int,
                        help='Kernel size of depth-wise conv in leff module')
    parser.add_argument('--leff_with_bn', default=True, help='Using batchnorm in leff module')

    return parser




'''ten-foldcross-validation'''
def main(args):
    best_metrics_all_folds = []
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    train_transform = build_transform(is_train=True, args=args)
    val_transform = build_transform(is_train=False, args=args)

    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0

    start_time = time.time()
    k = 10
    for i in range(0, k):
        print('*' * 25, 'The', i + 1, '-th fold', '*' * 25)

        train_data_set = MyDataSet(txt_path='dataset/train.txt', ki=i, K=k, typ='train',
                                   transform=train_transform, rand=True)

        valid_data_set = MyDataSet(txt_path='dataset/train.txt', ki=i, K=k, typ='val',
                                   transform=val_transform, rand=True)

        train_loader = DataLoader(train_data_set,
                                  shuffle=True,
                                  batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

        valid_loader = DataLoader(valid_data_set,
                                  shuffle=False,
                                  batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)

        print(f"Creating model: {args.model}")

        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
        )

        if args.finetune:  # None
            if args.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.finetune, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.finetune, map_location='cpu')
            if 'model' in checkpoint.keys():
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias',
                      'trans_cls_head.weight', 'trans_cls_head.bias', 'conv_cls_head.weight', 'conv_cls_head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            if 'pos_embed' in checkpoint_model.keys():
                # interpolate position embedding
                pos_embed_checkpoint = checkpoint_model['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                num_patches = model.patch_embed.num_patches
                num_extra_tokens = model.pos_embed.shape[-2] - num_patches
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

            model.load_state_dict(checkpoint_model, strict=False)

        model.to(device)
        model_ema = None
        if args.model_ema:  # True
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume='')

        model_without_ddp = model
        if args.distributed:  # False
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        # dummy_input = torch.randn(1, 3, args.input_size, args.input_size).to(device)
        # macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        # print(f"Params: {params / 1e6:.2f} M, FLOPs: {macs / 1e9:.2f} G")
        optimizer = create_optimizer(args, model)
        loss_scaler = NativeScaler()

        lr_scheduler, _ = create_scheduler(args, optimizer)

        criterion = LabelSmoothingCrossEntropy()

        if args.mixup > 0.:  # 0.8
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        output_dir = Path(args.output_dir)
        if args.resume:  # None
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            # pdb.set_trace()
            if 'model' in checkpoint.keys():
                model_without_ddp.load_state_dict(checkpoint['model'])
            else:
                model_without_ddp.load_state_dict(checkpoint)
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                if args.model_ema:
                    utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

        if args.eval:  #
            test_stats = evaluate(valid_loader, model, device)
            print(f"Accuracy of the network on the {len(valid_data_set)} test images: {test_stats['acc1']:.1f}%")
            return

        print("Start training")

        weights_name = "{}-{}-{}-({})-{}-{}.pth".format(args.model,args.input_size,args.lr, i + 1, args.mixup, args.cutmix)
        start_time = time.time()
        max_accuracy = 0.0
        best_metrics_this_fold = None
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model, criterion, train_loader,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
            )

            lr_scheduler.step(epoch)
            if args.output_dir:
                checkpoint_paths = [output_dir / weights_name]
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'args': args,
                    }, checkpoint_path)
            if epoch % args.evaluate_freq == 0:
                test_stats = evaluate(valid_loader, model, device)
                print(f"Accuracy of the network on the {len(valid_data_set)} test images: {test_stats['acc1']:.2f}%")
                acc_now = test_stats["acc1"]

                if acc_now > max_accuracy:
                    max_accuracy = acc_now
                    best_metrics_this_fold = [
                        acc_now,
                        test_stats["Pre"],
                        test_stats["Recall"],
                        test_stats["Spec"],
                        test_stats["F1"],
                        test_stats["Kappa"],
                        test_stats["Youden"],
                        test_stats["AUC"],
                    ]
                print(f'Max accuracy: {max_accuracy:.2f}%')

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}
                # "log-{}-({}).txt".format(args.model, i + 1)
                if args.output_dir and utils.is_main_process():
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")
        best_metrics_all_folds.append(best_metrics_this_fold)
        print(f"[Fold {i + 1}] best acc={max_accuracy:.3f}, metrics={best_metrics_this_fold}")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    best_metrics_all_folds = np.array(best_metrics_all_folds)  # shape = (10, 8)
    means = best_metrics_all_folds.mean(axis=0)
    stds = best_metrics_all_folds.std(axis=0, ddof=1)  

    metric_names = ["Acc", "Pre", "Recall", "Spec", "F1", "Kappa", "Youden", "AUC"]
    print("\n=== 10-fold results (mean ± std) ===")
    for name, m, s in zip(metric_names, means, stds):
        print(f"{name:7s}: {m:.4f} ± {s:.4f}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)