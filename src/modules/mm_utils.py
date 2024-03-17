import torch
import random

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from .dist_utils import all_gather
from ..gadgets.my_metrics import Accuracy, VQAScore, Scalar, F1


def set_metrics(pl_module):
    for split in ["train", "val", "test"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            setattr(pl_module, f"{split}_{k}_loss", Scalar())
            setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
            setattr(pl_module, f"{split}_{k}_f1", F1())
            

def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        if phase == "train":
            value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
            getattr(pl_module, f"train_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/train/f1_epoch",
                getattr(pl_module, f"train_{loss_name}_f1").compute()
            )
            getattr(pl_module, f"train_{loss_name}_f1").reset()
            pl_module.log(
                f"{loss_name}/train/loss_epoch",
                getattr(pl_module, f"train_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"train_{loss_name}_loss").reset()
        else:
            value = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/test/accuracy_epoch", value)
            getattr(pl_module, f"test_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/test/f1_epoch",
                getattr(pl_module, f"test_{loss_name}_f1").compute()
            )
            getattr(pl_module, f"test_{loss_name}_f1").reset()
            pl_module.log(
                f"{loss_name}/test/loss_epoch",
                getattr(pl_module, f"test_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"test_{loss_name}_loss").reset()

            value = getattr(pl_module, f"val_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/val/accuracy_epoch", value)
            getattr(pl_module, f"val_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/val/f1_epoch",
                getattr(pl_module, f"val_{loss_name}_f1").compute()
            )
            getattr(pl_module, f"val_{loss_name}_f1").reset()
            pl_module.log(
                f"{loss_name}/val/loss_epoch",
                getattr(pl_module, f"val_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"val_{loss_name}_loss").reset()
        the_metric = the_metric + value
        
    pl_module.log(f"{phase}/the_metric", the_metric)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return

def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.999)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )
