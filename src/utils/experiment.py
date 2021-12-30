import inspect
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.data import DataProcessor, DataSplitter, ExternalTestDataSplitter, get_loader
from utils.training import (
    GridSearch,
    df_to_array,
    external_test,
    grid_seach_on_val,
    sklearn_external_test,
    sklearn_test,
    test,
)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore


def generate_model_id():
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = os.path.basename(module.__file__)  # type: ignore
    filename_without_ext = os.path.splitext(filename)[0]
    return filename_without_ext


def run_experiment(
    model_id,
    data,
    model_class,
    loss_function,
    is_loss_extended,
    scorer,
    param_grid,
    config,
    **kwargs,
):
    ap_dicts = []

    seed_everything(2020)

    for i in range(1):
        split_indices = DataSplitter(
            binary_data=data,
            random_seed=i,
            test_delete_ratio=config["test_delete_ratio"],
        ).get_split_indices()

        processed_data = DataProcessor(
            binary_data=data,
            random_seed=i,
            test_delete_ratio=config["test_delete_ratio"],
        ).process(split_indices)

        train_loader, val_loader, test_loader = get_loader(
            processed_data, split_indices
        )

        best_model, best_param = GridSearch(
            model=model_class(data.shape[0], data.shape[1], config["k"]),
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            param_grid=param_grid,
            loss_function=loss_function,
            is_loss_extended=is_loss_extended,
            scorer=scorer,
            model_id=model_id,
            random_seed=i,
            **kwargs,
        ).search()

        test_ap_dict = test(best_model, test_loader, scorer)

        # grid search on the first run, and use the param for the rest
        param_grid = {}
        for k, v in best_param.items():
            param_grid[k] = [v]

        ap_dicts.append(test_ap_dict)

    return ap_dicts


def run_external_test(
    model_id,
    data_train,
    data_whole,
    model_class,
    loss_function,
    is_loss_extended,
    scorer,
    param_grid,
    config,
    **kwargs,
):
    ap_dicts = []

    seed_everything(2020)

    for i in range(1):
        split_indices = ExternalTestDataSplitter(
            binary_data=data_train,
            random_seed=i,
        ).get_split_indices()

        train_loader, val_loader = get_loader(data_train, split_indices)

        best_model, best_param = GridSearch(
            model=model_class(data_train.shape[0], data_train.shape[1], config["k"]),
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            param_grid=param_grid,
            loss_function=loss_function,
            is_loss_extended=is_loss_extended,
            scorer=scorer,
            model_id=model_id,
            random_seed=i,
            **kwargs,
        ).search()

        association_new = data_whole - data_train
        test_ap_dict = external_test(best_model, train_loader, association_new, scorer)

        # grid search on the first run, and use the param for the rest
        param_grid = {}
        for k, v in best_param.items():
            param_grid[k] = [v]

        ap_dicts.append(test_ap_dict)

    return ap_dicts


def _has_only_one_class(y: np.ndarray) -> bool:
    return len(np.unique(y)) == 1


def run_sklearn_experiment(
    data,
    severe_indices,
    model,
    param_grid,
    config,
):
    ap_dicts = []

    seed_everything(2020)

    for i in range(1):
        split_indices = DataSplitter(
            binary_data=data,
            random_seed=i,
            test_delete_ratio=config["test_delete_ratio"],
        ).get_split_indices()

        processed_data = DataProcessor(
            binary_data=data,
            random_seed=i,
            test_delete_ratio=config["test_delete_ratio"],
        ).process(split_indices)

        ap_dict = {}
        for i in tqdm(severe_indices):
            # convert data into sklear train format
            train_X, train_y, val_X, val_y, test_X, test_y = df_to_array(
                processed_data, split_indices, i
            )

            if _has_only_one_class(train_y):
                continue

            if _has_only_one_class(val_y):
                continue

            if _has_only_one_class(test_y):
                continue

            best_model = grid_seach_on_val(
                model, param_grid, train_X, train_y, val_X, val_y
            )

            score = sklearn_test(best_model, test_X, test_y)
            ap_dict[i] = score

        ap_dicts.append(ap_dict)

    return ap_dicts


def run_sklearn_external_test(
    data_train,
    data_whole,
    severe_indices,
    model,
    param_grid,
):
    ap_dicts = []

    seed_everything(2020)

    for i in range(1):
        split_indices = ExternalTestDataSplitter(
            binary_data=data_train,
            random_seed=i,
        ).get_split_indices()

        ap_dict = {}
        for i in tqdm(severe_indices):
            # convert data into sklear train format
            train_X, train_y, val_X, val_y = df_to_array(data_train, split_indices, i)

            association_new = data_whole - data_train
            _, train_y_new, _, _ = df_to_array(association_new, split_indices, i)

            if _has_only_one_class(train_y):
                continue

            if _has_only_one_class(val_y):
                continue

            if _has_only_one_class(train_y_new):
                continue

            best_model = grid_seach_on_val(
                model, param_grid, train_X, train_y, val_X, val_y
            )

            score = sklearn_external_test(best_model, train_X, train_y, train_y_new)

            if score is not None:
                ap_dict[i] = score

        ap_dicts.append(ap_dict)

    return ap_dicts


def save_results(model_id, ap_dicts):
    res_df = pd.DataFrame(ap_dicts)
    res_df["all"] = res_df.mean(axis=1)
    res_df.loc["mean", :] = res_df.mean(axis=0)  # type: ignore
    res_df.loc["std", :] = res_df.std(axis=0)  # type: ignore
    res_df.to_csv(f"./result/{model_id}.csv")
