import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 5, 6, 7"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from torch.utils.data import Subset
from functools import partial
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
from surgrec_video.augment.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from surgrec_video.core.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
import timm
from surgrec_video.data.datasets import build_dataset
from surgrec_video.engine.engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge
from surgrec_video.core.utils import NativeScalerWithGradNormCount as NativeScaler
from surgrec_video.core.utils import multiple_samples_collate
from surgrec_video.core import utils
import importlib




VARIANT_SPECS = {
    'dino': {
        'display_name': 'DINO',
        'modeling_module': 'surgrec_video.models.modeling_finetune_dino',
        'default_batch_size': 4,
        'default_model': 'dino_resnet50_patch16_224',
        'default_lr': 1e-3,
        'default_model_key': 'model|module|state_dict|student|teacher',
        'default_output_dir': 'output/dino_finetune',
        'resnet_like': True,
        'trainable_tokens': ('head', 'fc_norm', 'temporal_conv'),
        'extra_trainable_attrs': ('head', 'fc_norm', 'temporal_conv'),
    },
    'endofm': {
        'display_name': 'Endo-FM',
        'modeling_module': 'surgrec_video.models.modeling_finetune_EndoFM',
        'default_batch_size': 4,
        'default_model': 'endofm_vit_base_patch16_224',
        'default_lr': 1e-3,
        'default_model_key': 'model|module|state_dict|student|teacher',
        'default_output_dir': 'output/endofm_finetune',
        'resnet_like': False,
        'trainable_tokens': ('head',),
        'extra_trainable_attrs': ('head',),
    },
    'mocov2': {
        'display_name': 'MoCo V2',
        'modeling_module': 'surgrec_video.models.modeling_finetune_mocov2',
        'default_batch_size': 8,
        'default_model': 'mocov2_resnet50_patch16_224',
        'default_lr': 1e-3,
        'default_model_key': 'state_dict|model|encoder_q',
        'default_output_dir': 'output/mocov2_finetune',
        'resnet_like': True,
        'trainable_tokens': ('head', 'fc_norm', 'temporal_conv', 'temporal_bn'),
        'extra_trainable_attrs': ('head', 'fc_norm', 'temporal_conv', 'temporal_bn'),
    },
}


def get_variant_spec(name: str):
    key = name.lower()
    if key not in VARIANT_SPECS:
        raise ValueError(f'Unsupported backbone_variant: {name}')
    return VARIANT_SPECS[key]

def get_args():
    variant_parser = argparse.ArgumentParser(add_help=False)
    variant_parser.add_argument('--backbone_variant', default='dino', choices=sorted(VARIANT_SPECS.keys()))
    variant_args, _ = variant_parser.parse_known_args()
    variant = get_variant_spec(variant_args.backbone_variant)

    parser = argparse.ArgumentParser(
        f"{variant['display_name']} fine-tuning and evaluation script for video classification",
    )
    parser.add_argument('--backbone_variant', default=variant_args.backbone_variant, choices=sorted(VARIANT_SPECS.keys()),
                        help='Medical backbone variant to fine-tune')

    # Model and training parameters
    parser.add_argument('--batch_size', default=variant['default_batch_size'], type=int,
                        help='Batch size per GPU')
    parser.add_argument('--model', default=variant['default_model'], type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--lr', type=float, default=variant['default_lr'], metavar='LR',
                        help='learning rate')

    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs for fine-tuning (default: 50)')
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Video processing parameters
    parser.add_argument('--tubelet_size', type=int, default=2,
                        help='Tubelet size for temporal processing')
    parser.add_argument('--input_size', default=224, type=int, help='videos input size')

    # Dropout parameters
    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')

    # Model EMA parameters
    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.999])')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--layer_decay', type=float, default=0.85,
                        help='Layer decay rate for fine-tuning')

    # Learning rate schedule
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=1,
                        help='Repeated_aug (default: 1)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params for medical video
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, disabled for medical video (default: 0.0)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, disabled for medical video (default: 0.0)')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help=f"finetune from {variant['display_name']} checkpoint")
    parser.add_argument('--model_key', default=variant['default_model_key'], type=str,
                        help='Key for model weights in checkpoint')
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters for medical video
    parser.add_argument('--data_path', default='data/all', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=3, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames per video clip')
    parser.add_argument('--sampling_rate', type=int, default=4,
                        help='Temporal sampling rate')
    parser.add_argument('--data_set', default='all',
                        choices=['OOD_cataract-1k','OOD_cataract-101', 'Kinetics-400','OOD', 'SSV2', 'UCF101', 'HMDB51',
                                'image_folder', 'SurgKinetics','colonoscopic_web','endovis2019', 'JIGSAWS', 'cholec80',
                                'cholecT50', 'AutoLaparo', 'zju_phase', 'LDPolyVideo', 'AlxSuture', 'SurgicalActions160',
                                'all', 'kvasir-capsule', 'Hyper-kvasir', 'OOD_cat-21', 'M2CAI16-Workflow', 'SAR-RARP50',
                                'PitVis', 'MultiBypass140', 'LapGyn_dataset'],
                        type=str, help='dataset for medical backbone fine-tuning')

    # Output and logging
    parser.add_argument('--output_dir', default=variant['default_output_dir'],
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--frozen_backbone', action='store_true', default=False,
                        help='Whether to freeze backbone for fine-tuning')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test_num_workers', default=-1, type=int,
                        help='Number of workers for test DataLoader (default: -1 uses num_workers)')
    parser.add_argument('--pin_mem', default=True, action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def load_dino_checkpoint(model, checkpoint_path, args):
    """
    Load DINO pretrained checkpoint with proper key mapping for ResNet
    """
    if checkpoint_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location='cpu', check_hash=True)
    else:
        # Fix for PyTorch 2.6+ compatibility
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"Loading DINO checkpoint from {checkpoint_path}")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Handle DINO ClassyVision checkpoint format
    checkpoint_model = None
    
    # First try to find classy_state_dict (DINO specific)
    if 'classy_state_dict' in checkpoint:
        classy_dict = checkpoint['classy_state_dict']
        print(f"Found classy_state_dict with keys: {list(classy_dict.keys())}")
        
        # Look for base_model in classy_state_dict
        if 'base_model' in classy_dict:
            base_model = classy_dict['base_model']
            print(f"Found base_model with keys: {list(base_model.keys())}")
            
            # Look for model within base_model
            if 'model' in base_model:
                model_dict = base_model['model']
                print(f"Found model in base_model.model with sample keys: {list(model_dict.keys())[:5]}")
                
                # DINO checkpoint has 'trunk' and 'heads' in model
                if 'trunk' in model_dict:
                    checkpoint_model = model_dict['trunk']
                    print(f"Found trunk in base_model.model.trunk with sample keys: {list(checkpoint_model.keys())[:10]}")
                else:
                    checkpoint_model = model_dict
                    print(f"Using model dict directly with sample keys: {list(checkpoint_model.keys())[:5]}")
            elif 'trunk' in base_model:
                checkpoint_model = base_model['trunk']
                print(f"Found model in base_model.trunk with sample keys: {list(checkpoint_model.keys())[:5]}")
            else:
                checkpoint_model = base_model
                print(f"Using base_model directly with sample keys: {list(checkpoint_model.keys())[:5]}")
        else:
            # Look for other possible keys in classy_state_dict
            for possible_key in ['model', 'trunk', 'backbone']:
                if possible_key in classy_dict:
                    checkpoint_model = classy_dict[possible_key]
                    print(f"Found model weights in classy_state_dict[{possible_key}]")
                    break
    
    # Fallback to original method if classy_state_dict not found or empty
    if checkpoint_model is None:
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print(f"Found model weights with key: {model_key}")
                break
    
    if checkpoint_model is None:
        checkpoint_model = checkpoint
        print("Using checkpoint directly as model state dict")
    
    # Debug: Print sample keys from checkpoint_model
    if isinstance(checkpoint_model, dict):
        sample_keys = list(checkpoint_model.keys())[:10]
        print(f"Sample checkpoint model keys: {sample_keys}")
    
    # Get current model state dict for comparison
    state_dict = model.state_dict()
    model_keys = list(state_dict.keys())[:10]
    print(f"Sample model keys: {model_keys}")
    
    # Remove classification head if shape mismatch
    keys_to_remove = []
    for k in ['head.weight', 'head.bias', 'fc.weight', 'fc.bias', 'classifier.weight', 'classifier.bias']:
        if k in checkpoint_model and k in state_dict:
            if checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint due to shape mismatch: {checkpoint_model[k].shape} vs {state_dict[k].shape}")
                keys_to_remove.append(k)
    
    for k in keys_to_remove:
        del checkpoint_model[k]

    # More comprehensive key mapping for DINO ResNet checkpoint
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    
    # Track successful mappings
    mapped_keys = 0
    total_keys = len(all_keys)
    
    # Define multiple prefix removal strategies for DINO
    prefix_strategies = [
        '_feature_blocks.',  # DINO specific prefix
        'backbone.',         # Already has backbone prefix
        'encoder.',          # DINO specific  
        'module.backbone.',
        'module.encoder.',
        'module._feature_blocks.',
        'module.',
        'model.',
        'student.',
        'teacher.',
        ''  # No prefix removal
    ]
    
    for key in all_keys:
        mapped = False
        
        # Skip DINO specific components that don't exist in our model
        if any(skip_key in key for skip_key in ['head', 'neck', 'mlp', 'last_layer', 'prototypes']):
            print(f"Skipping DINO specific component: {key}")
            continue
        
        # Try direct mapping first (for weights that already have correct names)
        if key in state_dict:
            if checkpoint_model[key].shape == state_dict[key].shape:
                new_dict[key] = checkpoint_model[key]
                mapped_keys += 1
                mapped = True
                print(f"Direct mapping: {key}")
            else:
                print(f"Shape mismatch for direct mapping {key}: {checkpoint_model[key].shape} vs {state_dict[key].shape}")
        
        # Handle prefix removal and mapping to backbone
        if not mapped:
            for prefix in prefix_strategies:
                if key.startswith(prefix):
                    # Remove the prefix
                    new_key = key[len(prefix):]
                    
                    # Map to backbone structure
                    target_key = f'backbone.{new_key}'
                    
                    # Check if the mapped key exists in target model
                    if target_key in state_dict:
                        # Verify shape compatibility
                        if checkpoint_model[key].shape == state_dict[target_key].shape:
                            new_dict[target_key] = checkpoint_model[key]
                            mapped_keys += 1
                            mapped = True
                            print(f"Prefix mapping: {key} -> {target_key}")
                            break
                        else:
                            print(f"Shape mismatch for {key} -> {target_key}: {checkpoint_model[key].shape} vs {state_dict[target_key].shape}")
        
        # If still not mapped and not explicitly skipped, note it
        if not mapped:
            print(f"Could not map: {key}")
    
    print(f"Successfully mapped {mapped_keys}/{total_keys} keys")
    
    # If no keys were mapped, print debugging info
    if mapped_keys == 0:
        print("WARNING: No keys were successfully mapped!")
        print("Sample checkpoint keys:")
        for i, key in enumerate(list(checkpoint_model.keys())[:5]):
            print(f"  {key}")
        print("Sample model keys:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            print(f"  {key}")
        print("This suggests a significant architecture mismatch.")
        print("Proceeding with original checkpoint keys...")
        return checkpoint_model
    
    return new_dict



def load_endofm_checkpoint(model, checkpoint_path, args):
    """
    Load Endo-FM pretrained checkpoint with proper key mapping
    """
    if checkpoint_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location='cpu', check_hash=True)
    else:
        # Fix for PyTorch 2.6+ compatibility
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"Loading Endo-FM checkpoint from {checkpoint_path}")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Debug: Print some sample keys to understand the structure
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print(f"Found model weights with key: {model_key}")
            break
    
    if checkpoint_model is None:
        checkpoint_model = checkpoint
        print("Using checkpoint directly as model state dict")
    
    # Debug: Print sample keys from checkpoint_model
    if isinstance(checkpoint_model, dict):
        sample_keys = list(checkpoint_model.keys())[:10]
        print(f"Sample checkpoint model keys: {sample_keys}")
    
    # Get current model state dict for comparison
    state_dict = model.state_dict()
    model_keys = list(state_dict.keys())[:10]
    print(f"Sample model keys: {model_keys}")
    
    # Remove classification head if shape mismatch
    keys_to_remove = []
    for k in ['head.weight', 'head.bias', 'fc.weight', 'fc.bias', 'classifier.weight', 'classifier.bias']:
        if k in checkpoint_model and k in state_dict:
            if checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint due to shape mismatch: {checkpoint_model[k].shape} vs {state_dict[k].shape}")
                keys_to_remove.append(k)
    
    for k in keys_to_remove:
        del checkpoint_model[k]

    # More comprehensive key mapping for different checkpoint formats
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    
    # Track successful mappings
    mapped_keys = 0
    total_keys = len(all_keys)
    
    # Define multiple prefix removal strategies
    prefix_strategies = [
        'module.backbone.',
        'backbone.',
        'encoder.',
        'module.',
        'model.',
        'student.',
        'teacher.',
        ''  # No prefix removal
    ]
    
    for key in all_keys:
        mapped = False
        
        # Skip temporal attention components that don't exist in our model
        if any(temporal_key in key for temporal_key in ['temporal_norm1', 'temporal_attn', 'temporal_fc']):
            print(f"Skipping temporal component: {key}")
            continue
        
        # Skip position embeddings for now (they have different structures)
        if any(pos_key in key for pos_key in ['cls_token', 'pos_embed', 'time_embed']):
            print(f"Skipping position embedding: {key}")
            continue
        
        # Skip original head components
        if key.startswith('module.head.'):
            print(f"Skipping original head: {key}")
            continue
        
        # Skip norm layer that doesn't match
        if key == 'module.backbone.norm.weight' or key == 'module.backbone.norm.bias':
            # Map to fc_norm instead
            if 'norm.weight' in key:
                target_key = 'fc_norm.weight'
            else:
                target_key = 'fc_norm.bias'
            
            if target_key in state_dict:
                if checkpoint_model[key].shape == state_dict[target_key].shape:
                    new_dict[target_key] = checkpoint_model[key]
                    mapped_keys += 1
                    mapped = True
                    print(f"Mapped norm: {key} -> {target_key}")
            continue
        
        # Try different prefix removal strategies
        for prefix in prefix_strategies:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                
                # Check if the mapped key exists in target model
                if new_key in state_dict:
                    # Verify shape compatibility
                    if checkpoint_model[key].shape == state_dict[new_key].shape:
                        new_dict[new_key] = checkpoint_model[key]
                        mapped_keys += 1
                        mapped = True
                        break
                    else:
                        print(f"Shape mismatch for {key} -> {new_key}: {checkpoint_model[key].shape} vs {state_dict[new_key].shape}")
        
        # Special handling for attention bias - split qkv.bias into q_bias and v_bias
        if not mapped and 'attn.qkv.bias' in key:
            prefix_match = None
            for prefix in prefix_strategies:
                if key.startswith(prefix):
                    prefix_match = prefix
                    break
            
            if prefix_match:
                base_key = key[len(prefix_match):]
                q_bias_key = base_key.replace('attn.qkv.bias', 'attn.q_bias')
                v_bias_key = base_key.replace('attn.qkv.bias', 'attn.v_bias')
                
                if q_bias_key in state_dict and v_bias_key in state_dict:
                    qkv_bias = checkpoint_model[key]
                    if len(qkv_bias.shape) == 1:
                        dim = qkv_bias.shape[0] // 3
                        q_bias = qkv_bias[:dim]
                        v_bias = qkv_bias[2*dim:3*dim]  # Skip k_bias, only take q and v
                        
                        if q_bias.shape == state_dict[q_bias_key].shape and v_bias.shape == state_dict[v_bias_key].shape:
                            new_dict[q_bias_key] = q_bias
                            new_dict[v_bias_key] = v_bias
                            mapped_keys += 2
                            mapped = True
                            print(f"Split QKV bias: {key} -> {q_bias_key} + {v_bias_key}")
        
        # Special handling for patch embedding - convert 2D to 3D
        if not mapped and 'patch_embed.proj.weight' in key:
            prefix_match = None
            for prefix in prefix_strategies:
                if key.startswith(prefix):
                    prefix_match = prefix
                    break
            
            if prefix_match:
                base_key = key[len(prefix_match):]
                if base_key in state_dict:
                    checkpoint_tensor = checkpoint_model[key]
                    model_tensor = state_dict[base_key]
                    
                    # Handle 2D->3D conversion for patch embedding
                    if len(checkpoint_tensor.shape) == 4 and len(model_tensor.shape) == 5:
                        # checkpoint: [embed_dim, in_chans, patch_h, patch_w]
                        # model: [embed_dim, in_chans, tubelet_size, patch_h, patch_w]
                        print(f"Converting 2D patch embedding to 3D: {checkpoint_tensor.shape} -> {model_tensor.shape}")
                        
                        # Repeat along temporal dimension and average
                        converted_tensor = checkpoint_tensor.unsqueeze(2).repeat(1, 1, model_tensor.shape[2], 1, 1)
                        converted_tensor = converted_tensor / model_tensor.shape[2]
                        
                        if converted_tensor.shape == model_tensor.shape:
                            new_dict[base_key] = converted_tensor
                            mapped_keys += 1
                            mapped = True
                            print(f"Successfully converted patch embedding: {key} -> {base_key}")
        
        # If still not mapped and not explicitly skipped, note it
        if not mapped:
            print(f"Could not map: {key}")
    
    print(f"Successfully mapped {mapped_keys}/{total_keys} keys")
    
    # If no keys were mapped, print debugging info
    if mapped_keys == 0:
        print("WARNING: No keys were successfully mapped!")
        print("Sample checkpoint keys:")
        for i, key in enumerate(list(checkpoint_model.keys())[:5]):
            print(f"  {key}")
        print("Sample model keys:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            print(f"  {key}")
        print("This suggests a significant architecture mismatch.")
        print("Proceeding with original checkpoint keys...")
        return checkpoint_model
    
    checkpoint_model = new_dict

    # Handle position embedding interpolation
    if 'pos_embed' in checkpoint_model and 'pos_embed' in state_dict:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        pos_embed_model = state_dict['pos_embed']
        
        print(f"Position embedding shapes - checkpoint: {pos_embed_checkpoint.shape}, model: {pos_embed_model.shape}")
        
        if pos_embed_checkpoint.shape != pos_embed_model.shape:
            print("Interpolating position embeddings...")
            
            # Simple interpolation for now
            if len(pos_embed_checkpoint.shape) == 3 and len(pos_embed_model.shape) == 3:
                # [1, seq_len, embed_dim]
                B, N_old, C = pos_embed_checkpoint.shape
                _, N_new, _ = pos_embed_model.shape
                
                if N_old != N_new:
                    print(f"Position embedding interpolation: {N_old} -> {N_new}")
                    # Simple linear interpolation
                    pos_embed_checkpoint = pos_embed_checkpoint.permute(0, 2, 1)  # [1, C, N_old]
                    pos_embed_checkpoint = torch.nn.functional.interpolate(
                        pos_embed_checkpoint, size=N_new, mode='linear', align_corners=False)
                    pos_embed_checkpoint = pos_embed_checkpoint.permute(0, 2, 1)  # [1, N_new, C]
                    checkpoint_model['pos_embed'] = pos_embed_checkpoint

    return checkpoint_model



def load_mocov2_checkpoint(model, checkpoint_path, args):
    """
    Load MoCo V2 pretrained checkpoint with proper key mapping for ResNet
    MoCo V2 checkpoint format: uses 'state_dict' with 'encoder_q' prefix for backbone
    """
    if checkpoint_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location='cpu', check_hash=True)
    else:
        # Fix for PyTorch 2.6+ compatibility
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"Loading MoCo V2 checkpoint from {checkpoint_path}")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Handle MoCo V2 checkpoint format
    checkpoint_model = None
    
    # First check if this is a ClassyVision checkpoint (common for MoCo V2)
    if 'classy_state_dict' in checkpoint:
        classy_dict = checkpoint['classy_state_dict']
        print(f"Found classy_state_dict with keys: {list(classy_dict.keys())}")
        
        # Look for base_model in classy_state_dict
        if 'base_model' in classy_dict:
            base_model = classy_dict['base_model']
            print(f"Found base_model with keys: {list(base_model.keys())}")
            
            # Look for model within base_model
            if 'model' in base_model:
                model_dict = base_model['model']
                print(f"Found model in base_model.model with keys: {list(model_dict.keys())}")
                
                # MoCo V2 checkpoint typically has 'trunk' (backbone) and 'heads' (projection head)
                if 'trunk' in model_dict:
                    checkpoint_model = model_dict['trunk']
                    print(f"Found trunk in base_model.model.trunk with sample keys: {list(checkpoint_model.keys())[:10]}")
                else:
                    checkpoint_model = model_dict
                    print(f"Using model dict directly with sample keys: {list(checkpoint_model.keys())[:10]}")
            elif 'trunk' in base_model:
                checkpoint_model = base_model['trunk']
                print(f"Found model in base_model.trunk with sample keys: {list(checkpoint_model.keys())[:10]}")
            else:
                checkpoint_model = base_model
                print(f"Using base_model directly with sample keys: {list(checkpoint_model.keys())[:10]}")
        else:
            # Look for other possible keys in classy_state_dict
            for possible_key in ['model', 'trunk', 'backbone']:
                if possible_key in classy_dict:
                    checkpoint_model = classy_dict[possible_key]
                    print(f"Found model weights in classy_state_dict[{possible_key}]")
                    break
    
    # Fallback to standard MoCo V2 format only if ClassyVision parsing failed
    if checkpoint_model is None:
        print("ClassyVision parsing failed, trying standard MoCo V2 format...")
        if 'state_dict' in checkpoint:
            checkpoint_model = checkpoint['state_dict']
            print("Found 'state_dict' in checkpoint")
        elif 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
            print("Found 'model' in checkpoint")
        else:
            # This should not happen for proper checkpoints
            print("WARNING: Could not find model weights in expected locations!")
            print("Available checkpoint keys:", list(checkpoint.keys()))
            return None
    
    if checkpoint_model is None:
        raise RuntimeError("Cannot find model weights in checkpoint")
    
    # Debug: Print sample keys from checkpoint_model
    if isinstance(checkpoint_model, dict):
        sample_keys = list(checkpoint_model.keys())[:10]
        print(f"Sample checkpoint model keys: {sample_keys}")
    
    # Get current model state dict for comparison
    state_dict = model.state_dict()
    model_keys = list(state_dict.keys())[:10]
    print(f"Sample model keys: {model_keys}")
    
    # Remove classification head if shape mismatch (always remove for MoCo V2)
    keys_to_remove = []
    for k in ['head.weight', 'head.bias', 'fc.weight', 'fc.bias', 'classifier.weight', 'classifier.bias']:
        if k in checkpoint_model:
            print(f"Removing classifier key: {k}")
            keys_to_remove.append(k)
    
    for k in keys_to_remove:
        del checkpoint_model[k]

    # More comprehensive key mapping for MoCo V2 ResNet checkpoint
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    
    # Track successful mappings
    mapped_keys = 0
    total_keys = len(all_keys)
    
    # MoCo V2 specific prefixes - encoder_q contains the backbone
    # Also handle ClassyVision format prefixes
    prefix_strategies = [
        'encoder_q.',          # MoCo V2 query encoder (main backbone)
        'module.encoder_q.',   # Multi-GPU training
        '_feature_blocks.',    # ClassyVision format
        'module._feature_blocks.',  # Multi-GPU ClassyVision
        'backbone.',           # Some variations
        'module.backbone.',    # Multi-GPU backbone
        'module.',             # Standard multi-GPU prefix
        ''                     # No prefix
    ]
    
    for key in all_keys:
        mapped = False
        original_key = key
        
        # Try different prefix removal strategies
        for prefix in prefix_strategies:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                
                # Additional key mappings for ResNet
                if new_key.startswith('conv1.'):
                    target_key = f'backbone.{new_key}'
                elif new_key.startswith('bn1.'):
                    target_key = f'backbone.{new_key}'
                elif new_key.startswith('layer'):
                    target_key = f'backbone.{new_key}'
                elif new_key.startswith('avgpool'):
                    target_key = f'backbone.{new_key}'
                elif new_key.startswith('fc.') or new_key.startswith('classifier.'):
                    # Skip fully connected layer from MoCo V2
                    print(f"Skipping fc/classifier layer: {original_key}")
                    mapped = True
                    break
                else:
                    target_key = f'backbone.{new_key}'
                
                # Check if target key exists in model
                if target_key in state_dict:
                    # Check shape compatibility
                    if checkpoint_model[original_key].shape == state_dict[target_key].shape:
                        new_dict[target_key] = checkpoint_model[original_key]
                        mapped_keys += 1
                        mapped = True
                        print(f"Mapped: {original_key} -> {target_key}")
                        break
                    else:
                        print(f"Shape mismatch for {original_key} -> {target_key}: "
                              f"{checkpoint_model[original_key].shape} vs {state_dict[target_key].shape}")
        
        if not mapped:
            print(f"Failed to map key: {original_key}")
    
    print(f"Successfully mapped {mapped_keys}/{total_keys} keys")
    
    # If no keys were mapped, print debugging info
    if mapped_keys == 0:
        print("ERROR: No keys were successfully mapped!")
        print("This usually means the checkpoint format is different than expected.")
        print("Please check the checkpoint structure and key naming.")
        print("Proceeding with original checkpoint keys...")
        return checkpoint_model
    
    return new_dict




def get_checkpoint_loader(name: str):
    if name == 'dino':
        return load_dino_checkpoint
    if name == 'endofm':
        return load_endofm_checkpoint
    if name == 'mocov2':
        return load_mocov2_checkpoint
    raise ValueError(f'Unsupported backbone_variant: {name}')


def configure_model_window(args, model, variant):
    if variant['resnet_like']:
        print(f"Using ResNet-style {variant['display_name']} model")
        args.window_size = (
            args.num_frames // args.tubelet_size,
            args.input_size // 16,
            args.input_size // 16,
        )
        args.patch_size = (16, 16)
    else:
        patch_size = model.patch_embed.patch_size
        print("Patch size = %s" % str(patch_size))
        args.window_size = (
            args.num_frames // args.tubelet_size,
            args.input_size // patch_size[0],
            args.input_size // patch_size[1],
        )
        args.patch_size = patch_size


def freeze_backbone_parameters(model, variant):
    print("Freezing backbone parameters...")
    trainable_tokens = variant['trainable_tokens']
    for name, param in model.named_parameters():
        if not any(token in name for token in trainable_tokens):
            param.requires_grad = False

    for attr in variant['extra_trainable_attrs']:
        module = getattr(model, attr, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("Parameter Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
    print(f"  Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.1f}%)")
    print("Trainable components:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,} parameters")


def load_variant_checkpoint(model, args, variant):
    if not args.finetune:
        print("No pretrained model specified. Training from scratch.")
        return

    print(f"Loading {variant['display_name']} pretrained model from: {args.finetune}")
    checkpoint_model = get_checkpoint_loader(args.backbone_variant)(model, args.finetune, args)

    if checkpoint_model is None:
        print("Warning: checkpoint_model is None, skipping weight loading")
        return

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Missing keys: {len(missing_keys)}")
    if missing_keys:
        print(f"Missing keys (first 10): {missing_keys[:10]}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys (first 10): {unexpected_keys[:10]}")
    print(f"Successfully loaded {variant['display_name']} pretrained weights!")

def main(args, ds_init):
    variant = get_variant_spec(args.backbone_variant)
    importlib.import_module(variant['modeling_module'])

    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(f"{variant['display_name']} Fine-tuning Configuration:")
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Build datasets
    dataset_train, args.nb_classes = build_dataset(is_train=True, test_mode=False, args=args)
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)
    
    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Number of classes: {args.nb_classes}")
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    
    if args.dist_eval:
        if dataset_val is not None:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = None
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        test_num_workers = args.num_workers if args.test_num_workers is None or args.test_num_workers < 0 else args.test_num_workers
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=test_num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_test = None

    # Setup mixup (disabled for medical video by default)
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating {variant['display_name']} model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_checkpoint=args.use_checkpoint,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
    )

    configure_model_window(args, model, variant)
    load_variant_checkpoint(model, args, variant)

    model.to(device)

    if args.frozen_backbone:
        freeze_backbone_parameters(model, variant)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Training all parameters: {trainable_params:,} / {total_params:,}")
    
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Debug: Check parameter gradient status
    trainable_params = []
    frozen_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)
    
    print("Model = %s" % str(model_without_ddp))
    print('Number of trainable params:', n_parameters)
    print(f'Number of trainable parameter groups: {len(trainable_params)}')
    print(f'Number of frozen parameter groups: {len(frozen_params)}')
    if frozen_params:
        print("Frozen parameters:", frozen_params[:10], "..." if len(frozen_params) > 10 else "")
    
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    
    # Adjust learning rates
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    # Layer decay for fine-tuning
    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Using step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # Loss function
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    # Evaluation only mode
    if args.eval:
        preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
        test_stats = final_test(data_loader_test, model, device, preds_file, args.output_dir)
        torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
            print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats = {'Final top-1': final_top1,
                        'Final Top-5': final_top5}
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        exit(0)

    # Training loop
    print(f"Start {variant['display_name']} fine-tuning for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    
    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Training"):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
        )

        # Save checkpoint
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
                    
        # Validation
        if data_loader_val is not None:
            test_stats = validation_one_epoch(data_loader_val, model, device, epoch, args.output_dir)
            print(f"Accuracy of the network on the {len(dataset_val)} val videos: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(val_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(val_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(val_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # Final test
    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    test_stats = final_test(data_loader_test, model, device, preds_file, args.output_dir)
    torch.distributed.barrier()
    if global_rank == 0:
        print("Start merging results...")
        final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
        print(f"Final accuracy on {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        log_stats = {'Final top-1': final_top1,
                    'Final Top-5': final_top5}
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"{variant['display_name']} fine-tuning time: {total_time_str}")


if __name__ == '__main__':
    opts, ds_init = get_args()
    torch.cuda.empty_cache()
    
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)