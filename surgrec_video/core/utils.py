import builtins
import datetime
import io
import os
import time
import json
from pathlib import Path
from collections import defaultdict, deque

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.gpu)
    else:
        args.rank = 0
        args.gpu = 0
        args.distributed = False
        return
    args.distributed = True
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    import builtins as __builtins__
    builtin_print = __builtins__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtins__.print = print


class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{median:.4f} ({global_avg:.4f})'
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Synchronize the total and count across processes.
        The deque (window) is not synchronized; only global_avg will be correct.
        """
        if not is_dist_avail_and_initialized():
            return
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self.count = int(t[0].item())
        self.total = float(t[1].item())

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / max(1, self.count)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                '{}: {}'.format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def __getattr__(self, attr):
        # allow attribute-style access: logger.acc1 -> meters['acc1']
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"{self.__class__.__name__} has no attribute {attr}")

    def log_every(self, data_loader, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        for obj in data_loader:
            yield obj
            i += 1
            if i % print_freq == 0:
                print(header, self)
            end = time.time()
        total_time = time.time() - start_time
        print('{} Total time: {:.2f} s'.format(header, total_time))

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()


class TensorboardLogger:
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)

    def set_step(self, step=None):
        self.step = step

    def update(self, **kwargs):
        head = kwargs.pop('head', None)
        step = kwargs.pop('step', None)
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            tag = f'{head}/{k}' if head else k
            self.writer.add_scalar(tag, v, step if step is not None else self.step)

    def flush(self):
        self.writer.flush()


class NativeScalerWithGradNormCount:
    state_dict = lambda self: {"scale": 1.0}

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        loss.backward(create_graph=create_graph)
        grad_norm = None
        if clip_grad is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        if update_grad:
            optimizer.step()
            optimizer.zero_grad()
        return grad_norm


def save_model(args, model, model_without_ddp, optimizer, loss_scaler, epoch, model_ema=None):
    if not is_main_process():
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f'checkpoint-{epoch}.pth'
    torch.save({
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }, checkpoint_path)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema):
    resume_path = None
    if getattr(args, "resume", ""):
        if os.path.isfile(args.resume):
            resume_path = args.resume
        else:
            print(f"[WARN] Resume checkpoint not found: {args.resume}")
    if resume_path is None and getattr(args, "auto_resume", False) and getattr(args, "output_dir", None):
        output_dir = Path(args.output_dir)
        candidate = output_dir / "checkpoint-best.pth"
        if candidate.is_file():
            resume_path = str(candidate)
        else:
            checkpoints = sorted(output_dir.glob("checkpoint-*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if checkpoints:
                resume_path = str(checkpoints[0])
            else:
                candidate = output_dir / "checkpoint.pth"
                if candidate.is_file():
                    resume_path = str(candidate)

    if resume_path is None:
        return

    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)

    # Drop any mismatched keys (common when class counts differ)
    model_state = model_without_ddp.state_dict()
    for key in list(state_dict.keys()):
        if key not in model_state:
            continue
        if state_dict[key].shape != model_state[key].shape:
            del state_dict[key]

    msg = model_without_ddp.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Resume from {resume_path}")
    print(f"[INFO] Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")

    if not getattr(args, "eval", False) and optimizer is not None:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1


def load_state_dict(model, state_dict, prefix=''):
    model_state = model.state_dict()
    missing_keys = []
    unexpected_keys = []
    mismatched = []
    # Adapt 2D pretrained patch embedding weights to 3D (tubelet) if needed
    patch_key = f"{prefix}patch_embed.proj.weight" if prefix else "patch_embed.proj.weight"
    if patch_key in state_dict:
        target_state = model.state_dict()
        target_w = target_state.get(patch_key, None)
        source_w = state_dict[patch_key]
        if target_w is not None and source_w.ndim == 4 and target_w.ndim == 5:
            # Inflate 2D conv weights to 3D by repeating along temporal dimension
            # and normalizing to keep magnitude comparable.
            t = target_w.shape[2]
            inflated = source_w.unsqueeze(2).repeat(1, 1, t, 1, 1) / t
            state_dict[patch_key] = inflated
            if is_main_process():
                print(f"Inflated 2D patch_embed weights to 3D with tubelet_size={t}")

    # Convert qkv.bias -> q_bias/v_bias when model uses split biases
    target_state = model.state_dict()
    qkv_bias_key = f"{prefix}blocks.0.attn.qkv.bias" if prefix else "blocks.0.attn.qkv.bias"
    q_bias_key = f"{prefix}blocks.0.attn.q_bias" if prefix else "blocks.0.attn.q_bias"
    v_bias_key = f"{prefix}blocks.0.attn.v_bias" if prefix else "blocks.0.attn.v_bias"
    if qkv_bias_key in state_dict and q_bias_key in target_state and v_bias_key in target_state:
        if is_main_process():
            print("Converting qkv.bias to q_bias/v_bias for attention")
        new_state = dict(state_dict)
        for key in list(state_dict.keys()):
            if not key.endswith("attn.qkv.bias"):
                continue
            base = key[:-len("attn.qkv.bias")]
            q_key = f"{base}attn.q_bias"
            v_key = f"{base}attn.v_bias"
            if q_key not in target_state or v_key not in target_state:
                continue
            qkv_bias = state_dict[key]
            if qkv_bias.ndim == 1 and qkv_bias.numel() % 3 == 0:
                dim = qkv_bias.numel() // 3
                new_state[q_key] = qkv_bias[:dim].clone()
                new_state[v_key] = qkv_bias[-dim:].clone()
                new_state.pop(key, None)
        state_dict = new_state

    # Drop keys that are known to be incompatible with this model
    incompatible_prefixes = (
        "rope_embed.",
    )
    incompatible_suffixes = (
        "cls_token",
        "mask_token",
        "norm.weight",
        "norm.bias",
    )
    incompatible_contains = (
        ".gamma_1",
        ".gamma_2",
    )
    filtered = []
    if state_dict:
        for key in list(state_dict.keys()):
            key_no_prefix = key[len(prefix):] if prefix and key.startswith(prefix) else key
            if key_no_prefix.startswith(incompatible_prefixes):
                filtered.append(key)
                state_dict.pop(key, None)
                continue
            if key_no_prefix.endswith(incompatible_suffixes):
                filtered.append(key)
                state_dict.pop(key, None)
                continue
            if any(token in key_no_prefix for token in incompatible_contains):
                filtered.append(key)
                state_dict.pop(key, None)
        if filtered and is_main_process():
            print(f"Filtered incompatible keys: {len(filtered)}")

        filtered = {}

        for key, value in state_dict.items():
            mapped_key = key
            if prefix and mapped_key.startswith(prefix):
                mapped_key = mapped_key[len(prefix):]

            if mapped_key not in model_state:
                continue

            if model_state[mapped_key].shape != value.shape:
                mismatched.append((mapped_key, tuple(value.shape), tuple(model_state[mapped_key].shape)))
                continue

            filtered[mapped_key] = value

        missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)

        if is_main_process():
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
            if mismatched:
                print("Mismatched keys (ckpt -> model):")
                for name, ckpt_shape, model_shape in mismatched:
                    print(f"  {name}: {ckpt_shape} -> {model_shape}")

    if state_dict:
        merged_state = model_state.copy()
        merged_state.update(filtered)
        missing_keys, unexpected_keys = model.load_state_dict(merged_state, strict=True)
        if is_main_process():
            print(f"Missing keys after merge: {missing_keys}")
            print(f"Unexpected keys after merge: {unexpected_keys}")


def multiple_samples_collate(batch, fold=False):
    frames_list, labels_list, indexes_list, _ = zip(*batch)
    # flatten samples across num_sample
    frames = torch.stack([f for frames in frames_list for f in frames], dim=0)
    labels = torch.tensor([l for labels in labels_list for l in labels], dtype=torch.long)
    indexes = torch.tensor([i for idxs in indexes_list for i in idxs], dtype=torch.long)
    return frames, labels, indexes, {}


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, warmup_steps=-1):
    import numpy as np
    warmup_iters = warmup_epochs * niter_per_ep if warmup_steps < 0 else warmup_steps
    warmup_schedule = np.linspace(0, base_value, warmup_iters) if warmup_iters > 0 else np.array([])
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters))) if len(iters) > 0 else np.array([])
    schedule = np.concatenate((warmup_schedule, schedule)).astype(np.float32)
    return schedule
