from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module):
    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        raise NotImplementedError

    def loss_ingredients(
        self, user: Tensor, item: Tensor, **kwargs
    ) -> tuple[Tensor, ...]:
        """ingredients needed to compute loss besides model output"""
        raise NotImplementedError


class BaseLoss:
    def __call__(self, param: dict, pred: Tensor, t: Tensor, *args) -> Tensor:
        raise NotImplementedError
