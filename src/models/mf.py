import torch.nn as nn
from torch import Tensor

from utils.base import BaseLoss, BaseModel
from utils.data import Resource
from utils.experiment import generate_model_id, run_experiment, save_results
from utils.training import NumSevereSideEffects, Scorer


class MF(BaseModel):
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


class MFLoss(BaseLoss):
    def __call__(self, param: dict, pred: Tensor, t: Tensor, *args) -> Tensor:
        mse = nn.MSELoss(reduction="sum")
        return mse(pred, t)


if __name__ == "__main__":
    param_grid = {"embedding_l2": [0.0001, 0.0005, 0.001, 0.005, 0.01]}
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
        model_class=MF,
        loss_function=MFLoss(),
        is_loss_extended=False,
        scorer=scorer,
        param_grid=param_grid,
        config=config,
    )

    save_results(model_id, ap_dicts)
