from __future__ import annotations

import copy
import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def add_weight_decay(net, embedding_l2):
    embed, bias = [], []
    for name, param in net.named_parameters():
        if "bias" in name:
            bias.append(param)
        else:
            embed.append(param)
    return [
        {"params": embed, "weight_decay": embedding_l2},
        {"params": bias, "weight_decay": 0},
    ]


class EarlyStopping:
    def __init__(
        self,
        model_checkpoint_path,
        mode="min",
        patience=5,
        min_delta=0,
        percentage=False,
    ):
        self.model_checkpoint_path = model_checkpoint_path
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.prev_best = None
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False  # type: ignore

    def step(self, metric, model):
        if self.best is None:
            self.best = metric
            self._save_checkpoint(metric, model)
            return False

        if self.is_better(metric, self.best):  # type: ignore
            self.num_bad_epochs = 0
            self.prev_best = self.best
            self.best = metric
            self._save_checkpoint(metric, model)
        else:
            self.num_bad_epochs += 1
            print(f"Early stop count is {self.num_bad_epochs}")

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)

    def _save_checkpoint(self, metric, model):
        if self.prev_best is not None:
            print(
                f"Metric improved. ({self.prev_best:.6f} --> {self.best:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.model_checkpoint_path)


def _get_xy_from_loader(data, row_indices, column_indices, adr_index):
    X, y = [], []

    for ri, ci in zip(row_indices, column_indices):
        if ci == adr_index:
            X.append(np.delete(data[ri, :], ci))
            y.append(data[ri, ci])

    return np.array(X).reshape(-1, data.shape[1] - 1), np.array(y)


def df_to_array(
    data: np.ndarray,
    split_indices: tuple[np.ndarray, ...],
    adr_index: int,
) -> tuple[np.ndarray, ...]:
    """Convert binary data and split indices into X and y arrays for sklearn model training."""
    train_X, train_y = _get_xy_from_loader(
        data, split_indices[0], split_indices[1], adr_index
    )
    val_X, val_y = _get_xy_from_loader(
        data, split_indices[2], split_indices[3], adr_index
    )

    if len(split_indices) == 4:
        return train_X, train_y, val_X, val_y

    else:
        test_X, test_y = _get_xy_from_loader(
            data, split_indices[4], split_indices[5], adr_index
        )

        return train_X, train_y, val_X, val_y, test_X, test_y


def grid_seach_on_val(
    model: SVC,
    param_grid: dict,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> SVC:
    """Grid search on validation set to find best model."""

    best_score = 0
    best_model = None
    for param in ParameterGrid(param_grid):
        model = clone(model)  # type: ignore
        model.set_params(**param, random_state=42)

        model.fit(train_X, train_y)

        val_pred = model.predict_proba(val_X)[:, 1]

        score = average_precision_score(val_y, val_pred)

        if score > best_score:
            best_score = score
            best_model = model

    assert best_model is not None
    return best_model


class GridSearch:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        param_grid: dict,
        loss_function: Callable,
        is_loss_extended: bool,
        scorer: Callable,
        model_id: str,
        random_seed: int,
        **kwargs,
    ):
        self.model = model
        self.config = config
        self.param_grid = param_grid
        self.model_id = model_id
        self.scorer = scorer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.is_loss_extended = is_loss_extended
        self.random_seed = random_seed
        self.kwargs = kwargs

    def trial(self, suggested_param) -> float:
        model = copy.deepcopy(self.model).to(DEVICE)

        if "embedding_l2" in suggested_param:
            params = add_weight_decay(model, suggested_param["embedding_l2"])
        else:
            params = model.parameters()

        es = EarlyStopping(
            f"./models/{self.model_id}_{self.random_seed+1}.pt",
            mode="max",
            patience=self.config["es_patience"],
        )

        if self.config["optimizer_name"] == "adam":
            optimizer = optim.Adam(params, lr=self.config["lr"])
        else:
            raise ValueError("Not a valid optimizer name.")

        if self.config["scheduler_name"] == "onplateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.1,
                patience=self.config["sc_patience"],
                verbose=False,
            )
        else:
            raise ValueError("Not a valid scheduler name.")

        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config,
            param=suggested_param,
            loss_function=self.loss_function,
            is_loss_extended=self.is_loss_extended,
            scorer=self.scorer,
            random_seed=self.random_seed,
            **self.kwargs,
        )

        val_map = None
        for _ in range(trainer.config["epochs"]):
            trainer.train_step(optimizer)
            val_map = trainer.val_step(scheduler)

            if es.step(val_map, trainer.model):
                val_map = es.best
                break

        # if epochs are positive, val_map is not None
        assert val_map is not None

        return val_map

    def search(self) -> tuple[nn.Module, dict]:
        best_score = 0
        best_model = None
        best_param = None

        for suggested_param in ParameterGrid(self.param_grid):
            val_map = self.trial(suggested_param)

            # store if best result is obtained
            if val_map > best_score:
                os.rename(
                    f"./models/{self.model_id}_{self.random_seed+1}.pt",
                    f"./models/{self.model_id}_best_{self.random_seed+1}.pt",
                )
                self.model.load_state_dict(
                    torch.load(f"./models/{self.model_id}_best_{self.random_seed+1}.pt")
                )
                best_model = self.model.to(DEVICE)
                best_score = val_map
                best_param = suggested_param

        # if the param_grid is not empty, best_model and best_param are not None
        assert best_param is not None
        assert best_model is not None

        return best_model, best_param


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        param,
        loss_function: Callable,
        is_loss_extended: bool,
        scorer: Callable,
        random_seed: int,
        **kwargs,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.param = param
        self.loss_function = loss_function
        self.is_loss_extended = is_loss_extended
        self.scorer = scorer
        self.random_seed = random_seed
        self.kwargs = kwargs

    def train_step(self, optimizer):
        for i, c, t in self.train_loader:
            self.model.zero_grad()
            i, c, t = i.to(DEVICE), c.to(DEVICE), t.to(DEVICE)
            pred = self.model(i, c)

            if self.is_loss_extended:
                loss_ingredients = self.model.loss_ingredients(i, c, **self.kwargs)
                loss = self.loss_function(
                    self.param, pred.reshape(-1), t.float(), *loss_ingredients
                )
            else:
                loss = self.loss_function(self.param, pred.reshape(-1), t.float())

            loss.backward()
            optimizer.step()

    def val_step(self, scheduler) -> float:
        adr_idx = []
        val_pred = []
        val_true = []
        for i, c, t in self.val_loader:
            self.model.eval()
            i, c, t = i.to(DEVICE), c.to(DEVICE), t.to(DEVICE)
            pred = self.model(i, c)

            adr_idx.append(c.detach().cpu())
            val_pred.append(pred.detach().cpu())
            val_true.append(t.detach().cpu())

        adr_idx = torch.cat(adr_idx).numpy()
        val_pred = torch.cat(val_pred).numpy()
        val_true = torch.cat(val_true).numpy()

        ap_dict = self.scorer(adr_idx, val_pred, val_true)
        severe_mean_ap = sum(ap_dict.values()) / len(ap_dict.values())
        scheduler.step(severe_mean_ap)
        return severe_mean_ap


class NumSevereSideEffects:
    FAERS = 68
    SIDER = 70


class Scorer:
    def __init__(self, num_side_effects: int, num_severe_side_effects: int):
        self.num_side_effects = num_side_effects
        self.num_severe_side_effects = num_severe_side_effects

    def __call__(self, adr_idx, val_pred, val_true):
        severe_indices = np.arange(self.num_side_effects)[
            -self.num_severe_side_effects :  # noqa
        ]
        ap_dict = {}
        for c in severe_indices:
            # skip if no positive example for the adr
            if sum(val_true[adr_idx == c]) == 0:
                continue
            ap = average_precision_score(val_true[adr_idx == c], val_pred[adr_idx == c])
            ap_dict[c] = ap

        return ap_dict


def test(best_model, test_loader, scorer):
    adr_idx = []
    test_pred = []
    test_true = []
    for i, c, t in test_loader:
        best_model.eval()
        i, c, t = i.to(DEVICE), c.to(DEVICE), t.to(DEVICE)
        pred = best_model(i, c)

        adr_idx.append(c.detach().cpu())
        test_pred.append(pred.detach().cpu())
        test_true.append(t.detach().cpu())

    adr_idx = torch.cat(adr_idx).numpy()
    test_pred = torch.cat(test_pred).numpy()
    test_true = torch.cat(test_true).numpy()

    ap_dict = scorer(adr_idx, test_pred, test_true)
    return ap_dict


def external_test(best_model, train_loader, association_new, scorer):
    """test prediction performance on negative data (possibly positive) in train."""
    adr_idx = []
    test_pred = []
    test_true = []
    for i, c, t in train_loader:
        best_model.eval()
        i, c, t = i.to(DEVICE), c.to(DEVICE), t.to(DEVICE)
        pred = best_model(i, c)

        adr_idx.append(c[t == 0].detach().cpu())
        test_pred.append(pred[t == 0].detach().cpu())
        test_true.append(
            torch.from_numpy(
                association_new[
                    i[t == 0].detach().cpu().numpy(),
                    c[t == 0].detach().cpu().numpy(),
                ]
            )
        )

    adr_idx = torch.cat(adr_idx).numpy()
    test_pred = torch.cat(test_pred).numpy()
    test_true = torch.cat(test_true).numpy()

    ap_dict = scorer(adr_idx, test_pred, test_true)
    return ap_dict


def sklearn_test(best_model, test_X, test_y) -> float:
    test_pred = best_model.predict_proba(test_X)[:, 1]

    score = average_precision_score(test_y, test_pred)
    return score  # type: ignore


def sklearn_external_test(best_model, test_X, test_y, test_y_new) -> float:
    test_pred = best_model.predict_proba(test_X)[:, 1]

    score = average_precision_score(test_y_new[test_y == 0], test_pred[test_y == 0])
    return score  # type: ignore
