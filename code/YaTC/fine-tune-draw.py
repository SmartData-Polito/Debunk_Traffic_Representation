import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import os
import PIL
import torch
import torch.nn as nn
from functools import partial
from torchvision import datasets, transforms

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_YaTC

from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('YaTC fine-tuning for traffic classification', add_help=False)
    # 64
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='TraFormer_YaTC', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=40, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
#20
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--finetune', default='./output_dir/pretrained-model.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--data_path', default='./data/ISCXVPN2016_MFR', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def build_dataset(type, args):
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if type == 'test':
        dataset = datasets.ImageFolder(f"{args.test_path}", transform=transform)
    else:
        dataset = datasets.ImageFolder(f"{args.data_path}/{type}", transform=transform)
    return dataset

class TrafficTransformerPre(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(TrafficTransformerPre, self).__init__(**kwargs)

        self.patch_embed = models_YaTC.PatchEmbed(img_size=kwargs['img_size'], patch_size=kwargs['patch_size'],
                                         in_chans=kwargs['in_chans'], embed_dim=kwargs['embed_dim'])

        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        self.fc_norm = norm_layer(embed_dim)
        del self.norm  # remove the original norm

    def forward_packet_features(self, x, i):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        cls_pos = self.pos_embed[:, :1, :]
        packet_pos = self.pos_embed[:, i*80+1:i*80+81, :]
        pos_all = torch.cat((cls_pos, packet_pos), dim=1)
        x = x + pos_all
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        cls = x[:, :1, :]

        x = x[:, 1:, :]
        x = x.reshape(B, 4, 20, -1).mean(axis=1)
        x = torch.cat((cls, x), dim=1)

        self.fc_norm(x)

        return x

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, 5, -1)
        for i in range(5):
            packet_x = x[:, :, i, :]

            packet_x = packet_x.reshape(B, C, -1, 40)
            packet_x = self.forward_packet_features(packet_x, i)

            if i == 0:
                new_x = packet_x
            else:
                new_x = torch.cat((new_x, packet_x), dim=1)
        x = new_x

        for blk in self.blocks:
            x = blk(x)

        x = x.reshape(B, 5, 21, -1)[:, :, 0, :]
        x = x.mean(dim=1)
        
        outcome = self.fc_norm(x)
        # outcome = outcome.view(outcome.shape[0],1,-1)

        return outcome

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)

        return x
    
@torch.no_grad()
def evaluatePre(data_loader, model, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()
    outputs = []
    targets = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output

        with torch.cuda.amp.autocast():
            output = model(images).cpu().numpy()
            outputs.extend(output.tolist())
            targets.extend(target.cpu().numpy().tolist())

    return outputs, targets

def main(args):
    # misc.init_distributed_mode(args)
    args.distributed = False

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(type = "train", args=args)
    dataset_val = build_dataset(type = "val", args=args)
    dataset_test = build_dataset(type = "test", args=args)

    labels = dataset_val.classes

    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = TrafficTransformerPre(img_size=40, patch_size=2, in_chans=1, embed_dim=192, depth=4, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=args.nb_classes, drop_path_rate=args.drop_path,)

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
        print("Model has, but not in parameters:", missing_keys)
        print("Parameters have, but not in model:", unexpected_keys)

        # 冻结嵌入层和预训练层的参数
        if args.frozen:
            print("Froze the embedding and pre-training layer")
            for param in model.parameters():
                param.requires_grad = False  # 冻结所有参数

            # 解冻特定的层
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
            model.fc_norm.weight.requires_grad = True
            model.fc_norm.bias.requires_grad = True

        # 遍历模型参数，打印其冻结状态
        print("\nTrainable Parameters:")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            print(f"Layer: {name} | Frozen: {not param.requires_grad}")

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified，before is 256, dont know why
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)


    embedding_outputs, targets_outputs = evaluatePre(data_loader_test, model, device)
    print("len(embedding_outputs):", len(embedding_outputs), "len(targets_outputs):", len(targets_outputs))

    os.makedirs(f"results/{args.dataset}", exist_ok=True)
    with open(f"results/{args.dataset}/{args.data_path[-1]}_{args.blr}_{args.frozen}_embedding_outputs.json", "w") as f:
        json.dump(embedding_outputs, f)

    with open(f"results/{args.dataset}/{args.data_path[-1]}_{args.blr}_{args.frozen}_targets_outputs.json", "w") as f:
        json.dump(targets_outputs, f)

    

if __name__ == '__main__':
    args = get_args_parser()
    args.add_argument("--frozen", action="store_true", help="Whether frozens the embedding and pre-training layer")
    args.add_argument("--test_path", type=str, required=True, help="Specify the dataset used for training.")
    args.add_argument("--dataset", type=str, required=True, help="Specify the dataset used for training.")

    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
