from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class LossFunction(str, Enum):
    l1 = nn.L1Loss
    mse = nn.MSELoss

class Optimizer(str, Enum):
    adam = torch.optim.Adam
    sgd = torch.optim.SGD


@dataclass
class Config:
    epochs: int
    lr: float
    batch_size: int
    # slice windows
    data_path: str
    run_dir: str
    device: str = 'cuda:1'
    loss_function: LossFunction = LossFunction.mse
    optimizer : Optimizer = Optimizer.adam
    # Loss weights for generalized loss function
    loss_alpha: float = 0.5  # Weight for spatial loss
    loss_beta: float = 0.3   # Weight for count loss  