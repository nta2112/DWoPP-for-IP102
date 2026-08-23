import torch
import torch.nn as nn
import os
import glob
from typing import Optional, List, Tuple


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def get_loader_kwargs(batch_size: int, num_gpus: int, drop_last: bool = True) -> dict:
    if num_gpus > 1:
        if batch_size % num_gpus != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by num_gpus ({num_gpus})")
        return {'batch_size': batch_size, 'drop_last': drop_last}
    return {'batch_size': batch_size, 'drop_last': drop_last}


def find_data_root(local_paths: List[str], env_var: str = None, kaggle_base: str = '/kaggle/input') -> str:
    if env_var and os.environ.get(env_var):
        path = os.environ[env_var]
        if os.path.exists(path):
            return path
    
    for path in local_paths:
        if os.path.exists(path):
            return path
    
    kaggle_inputs = glob.glob(os.path.join(kaggle_base, '*'))
    for kp in kaggle_inputs:
        for lp in local_paths:
            candidate = os.path.join(kp, os.path.basename(lp))
            if os.path.exists(candidate):
                return candidate
            if os.path.exists(kp) and any(fname.endswith('.json') for fname in os.listdir(kp) if os.path.isfile(os.path.join(kp, fname))):
                return kp
    
    raise FileNotFoundError(f"Could not find dataset. Tried: {local_paths}, env: {env_var}, kaggle: {kaggle_base}")


def find_file_deep(filename: str, search_dirs: List[str]) -> str:
    for base in search_dirs:
        for root, dirs, files in os.walk(base):
            if filename in files:
                return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found in {search_dirs}")


def get_gpu_ids(gpu_arg: str) -> List[int]:
    if not gpu_arg:
        return []
    gpus = ''.join(gpu_arg.split())
    return [int(gid) for gid in gpus.split(',')]


def auto_detect_gpus() -> List[int]:
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []