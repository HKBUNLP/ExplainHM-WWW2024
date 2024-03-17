import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from .dist_utils import all_gather

SMALL_NUM = np.log(1e-45)

def compute_clf(pl_module, ret):
    logits = ret["logits"]
    label_ids = ret["label_ids"]
    clf_loss = F.cross_entropy(logits, label_ids.view(-1))

    new_ret = {
        f"clf_loss": clf_loss
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_clf_loss")(new_ret["clf_loss"])
    acc = getattr(pl_module, f"{phase}_clf_accuracy")(
        ret["logits"], ret["label_ids"]
    )
    f1 = getattr(pl_module, f"{phase}_clf_f1")(
        ret["logits"], ret["label_ids"]
    )
    pl_module.log(f"clf/{phase}/loss", loss)
    pl_module.log(f"clf/{phase}/accuracy", acc)
    pl_module.log(f"clf/{phase}/f1", f1)
    return new_ret


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()