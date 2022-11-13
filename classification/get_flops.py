import argparse
from typing import NoReturn
from functools import partial
import torch
import torch.nn as nn

from timm.models import create_model
from torch.nn.modules.normalization import LayerNorm

from tools import get_model_complexity_info, ln_flops_counter_hook
import models

def get_args_parser():
    parser = argparse.ArgumentParser('FLOPs counter', add_help=False)
    parser.add_argument('--shape', default=[224,224], type=int)
    parser.add_argument('--model', default='L2ViT_Tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--batch-size', default=64, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')       
    # distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    return parser

def main(args):
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    torch.manual_seed(0)
    model = create_model(
        args.model, 
        pretrained=False, 
        num_classes=1000, 
        drop_path_rate=0,
        )

    model.eval()
    model.cuda()
    # copy from uniformer
    # from fvcore.nn import FlopCountAnalysis
    # from fvcore.nn import flop_count_table
    # flops = FlopCountAnalysis(model, torch.ones((1, *input_shape), 
    #                                          dtype=next(model.parameters()).dtype,
    #                                          device=next(model.parameters()).device))
    # print(flop_count_table(flops))

    custom_modules_hooks = {}
    for name, module in model.named_modules():
        if hasattr(module, 'custom_modules_hooks'):
            custom_modules_hooks = module.custom_modules_hooks
            break
    
    flops, params = get_model_complexity_info(model, input_shape, custom_modules_hooks=custom_modules_hooks)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')  

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FLOPs counter', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
