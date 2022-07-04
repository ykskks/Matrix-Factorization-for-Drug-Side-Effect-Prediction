from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from utils.base import BaseLoss, BaseModel
from utils.data import Resource
from utils.experiment import generate_model_id, run_experiment, save_results
from utils.training import NumSevereSideEffects, Scorer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MFL1(BaseModel):
    def __init__(self, n_user: int, n_item: int, k: int):
        super().__init__()
        self.user_factors = nn.Embedding(n_user, k)
        self.item_factors = nn.Embedding(n_item, k)
        self.user_biases = nn.Embedding(n_user, 1)
        self.item_biases = nn.Embedding(n_item, 1)

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        out = self.user_biases(user) + self.item_biases(item)
        out += (self.user_factors(user) * self.item_factors(item)).sum(1, keepdim=True)
        return out

    def loss_ingredients(
        self, user: Tensor, item: Tensor, **kwargs
    ) -> tuple[Tensor, ...]:
        return (self,)  # type: ignore


class MFL1Loss(BaseLoss):
    def __call__(self, param: dict, pred: Tensor, t: Tensor, *args) -> Tensor:
        model: nn.Module = args[0]

        mse = nn.MSELoss(reduction="sum")
        l1 = nn.L1Loss(reduction="sum")
        l1_reg = 0

        for n, p in model.named_parameters():
            if "bias" not in n:
                l1_reg += l1(p, torch.zeros(p.data.shape).to(DEVICE))

        return mse(pred, t) + param["embedding_l1"] * l1_reg


if __name__ == "__main__":
    param_grid = {
        "embedding_l1": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    }
    config = {
        "k": 100,
        "epochs": 100,
        "lr": 0.01,
        "optimizer_name": "adam",
        "scheduler_name": "onplateau",
        "es_patience": 2,
        "sc_patience": 0,
        "test_delete_ratio": 0,
    }

    model_id = generate_model_id()

    resource = Resource("./data")
    data = resource.load_faers_train(threshold=3)
    scorer = Scorer(data.shape[1], NumSevereSideEffects.FAERS)

    ap_dicts = run_experiment(
        model_id=model_id,
        data=data,
        model_class=MFL1,
        loss_function=MFL1Loss(),
        is_loss_extended=True,
        scorer=scorer,
        param_grid=param_grid,
        config=config,
    )

    save_results(model_id, ap_dicts)
