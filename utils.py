import json
import math
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def adjust_learning_rate(optimizer, epoch: int, args) -> None:
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))
        }
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    else:
        lr_adjust = {}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")


class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss: float, model: torch.nn.Module, checkpoint_path: str) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_path)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return

        self.best_score = score
        self.save_checkpoint(val_loss, model, checkpoint_path)
        self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module, checkpoint_path: str) -> None:
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saving model."
            )
        torch.save(model.state_dict(), checkpoint_path)
        self.val_loss_min = val_loss
