from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from utils.base import BaseLoss, BaseModel
from utils.data import Resource
from utils.experiment import generate_model_id, run_experiment, save_results
from utils.training import NumSevereSideEffects, Scorer


class FGRMF(BaseModel):
    """
    Feature-derived graph reguralized MF
    """

    def __init__(self, n_user: int, n_item: int, k: int):
        super().__init__()
        self.user_factors = nn.Embedding(n_user, k)
        self.item_factors = nn.Embedding(n_item, k)

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

    def loss_ingredients(
        self, user: Tensor, item: Tensor, **kwargs
    ) -> tuple[Tensor, ...]:
        # generate pairs of drug indices in a batch
        comb = torch.combinations(item, 2)
        first_indices = comb[:, 0]
        sec_indices = comb[:, 1]

        sim_matrix = kwargs.pop("sim_matrix")
        sim = sim_matrix[first_indices, sec_indices].double()

        diff = self.user_factors(first_indices) - self.user_factors(sec_indices)
        diff_norm = torch.norm(diff, p="fro", dim=1).double()
        return sim, diff_norm


class FGRMFLoss(BaseLoss):
    def __call__(self, param: dict, pred: Tensor, t: Tensor, *args) -> Tensor:
        sim, diff_norm = args[0], args[1]
        mse = nn.MSELoss(reduction="sum")
        sim_reg = torch.dot(sim, diff_norm)
        return mse(pred, t) + param["sim_lambda"] * sim_reg


if __name__ == "__main__":
    param_grid = {
        "embedding_l2": [0.0001, 0.0005, 0.001, 0.005, 0.01],
        "sim_lambda": [0.00001, 0.00005, 0.0001, 0.0005, 0.001],
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
        model_class=FGRMF,
        loss_function=FGRMFLoss(),
        is_loss_extended=True,
        scorer=scorer,
        param_grid=param_grid,
        config=config,
        # loss_ingredients_kwargs
        sim_matrix=resource.load_sim_matrix(),
    )

    save_results(model_id, ap_dicts)
