from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Resource:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_faers_train(self, threshold: int) -> np.ndarray:
        """returns faers association data in train period"""

        count = pd.read_csv(
            self.data_dir / "crosstab_train.csv", index_col="keggid_new"
        )
        # binarize target: count -> association
        association = (count >= threshold).astype(int)
        return association.values

    def load_faers_whole(self, threshold: int) -> np.ndarray:
        """returns faers association data in train & test period"""

        count_train = pd.read_csv(
            self.data_dir / "crosstab_train.csv", index_col="keggid_new"
        )
        count_test = pd.read_csv(
            self.data_dir / "crosstab_test.csv", index_col="keggid_new"
        )
        # binarize target: count -> association
        association = ((count_train + count_test) >= threshold).astype(int)
        return association.values

    def load_sider(self) -> np.ndarray:
        """returns sider association data"""
        return pd.read_csv(
            self.data_dir / "sider_association.csv", index_col="keggid_new"
        ).values

    def load_C(self) -> Tensor:
        """returns count data to be used in LogisticMF"""
        count = pd.read_csv(
            self.data_dir / "crosstab_train.csv", index_col="keggid_new"
        ).values
        count = count - 2
        count[count <= 0] = 0
        C = torch.tensor(count, device=DEVICE).float()
        return C

    def load_sim_matrix(self) -> Tensor:
        """returns similarity matrix to be used in FGRMF"""
        return Tensor(np.load(self.data_dir / "pubchem_sim_matrix.npy"), device=DEVICE)


class DataSplitter:
    def __init__(
        self, binary_data: np.ndarray, random_seed: int, test_delete_ratio: float
    ):
        self.binary_data = binary_data
        self.random_seed = random_seed
        self.test_delete_ratio = test_delete_ratio

    def _split_by_drugs(self, raw_data, ri, ci):
        drug_indices = np.arange(raw_data.shape[0])
        _, test_drug_indices = train_test_split(
            drug_indices, test_size=0.5, shuffle=True, random_state=self.random_seed
        )
        test_ci, test_ri = (
            ci[np.isin(ri, test_drug_indices)],
            ri[np.isin(ri, test_drug_indices)],
        )
        train_ci, train_ri = (
            ci[~np.isin(ri, test_drug_indices)],
            ri[~np.isin(ri, test_drug_indices)],
        )
        return train_ri, train_ci, test_ri, test_ci

    def _split_by_entries(self, raw_data, train_ri, train_ci, test_ri, test_ci):
        train_is_pos = (raw_data[train_ri, train_ci].reshape(-1) == 1).astype(int)
        test_is_pos = (raw_data[test_ri, test_ci].reshape(-1) == 1).astype(int)
        train_idx_1, val_idx = train_test_split(
            np.arange(len(train_ri)),
            test_size=0.2,
            stratify=train_is_pos,
            random_state=self.random_seed,
        )
        train_idx_2, test_idx = train_test_split(
            np.arange(len(test_ri)),
            test_size=0.4,
            stratify=test_is_pos,
            random_state=self.random_seed,
        )

        train_ri_1, train_ci_1 = train_ri[train_idx_1], train_ci[train_idx_1]
        train_ri_2, train_ci_2 = test_ri[train_idx_2], test_ci[train_idx_2]
        val_ri, val_ci = train_ri[val_idx], train_ci[val_idx]
        test_ri, test_ci = test_ri[test_idx], test_ci[test_idx]

        train_ri = np.concatenate([train_ri_1, train_ri_2], axis=0)
        train_ci = np.concatenate([train_ci_1, train_ci_2], axis=0)

        split_indices = (train_ri, train_ci, val_ri, val_ci, test_ri, test_ci)
        return split_indices

    def get_split_indices(self):
        # deepcopy to avoid unintentional change to original data
        raw_data = copy.deepcopy(self.binary_data)

        # get row index and column index of each entry
        ri = np.arange(raw_data.shape[0] * raw_data.shape[1]) // raw_data.shape[1]
        ci = np.arange(raw_data.shape[0] * raw_data.shape[1]) % raw_data.shape[1]

        # split drugs -> train drugs and test drugs
        train_ri, train_ci, test_ri, test_ci = self._split_by_drugs(raw_data, ri, ci)

        # split entries
        # train drugs -> train: val = 0.8: 0.2
        # test drugs -> train: test = 0.6: 0.4
        split_indices = self._split_by_entries(
            raw_data, train_ri, train_ci, test_ri, test_ci
        )
        return split_indices


class ExternalTestDataSplitter:
    def __init__(self, binary_data: np.ndarray, random_seed: int):
        self.binary_data = binary_data
        self.random_seed = random_seed

    def get_split_indices(self):
        # deepcopy to avoid unintentional change to original data
        raw_data = copy.deepcopy(self.binary_data)

        # get row index and column index of each entry
        ri = np.arange(raw_data.shape[0] * raw_data.shape[1]) // raw_data.shape[1]
        ci = np.arange(raw_data.shape[0] * raw_data.shape[1]) % raw_data.shape[1]

        # split entries
        # all train entries -> train: val = 0.9: 0.1
        train_indices, valid_indices = train_test_split(
            range(len(ri)), test_size=0.1, shuffle=True, random_state=self.random_seed
        )
        train_ri, train_ci = ri[train_indices], ci[train_indices]
        val_ri, val_ci = ri[valid_indices], ci[valid_indices]

        self.processed_data = self.binary_data
        split_indices = (train_ri, train_ci, val_ri, val_ci)
        return split_indices


class DataProcessor:
    def __init__(
        self, binary_data: np.ndarray, random_seed: int, test_delete_ratio: float
    ):
        self.binary_data = binary_data
        self.random_seed = random_seed
        self.test_delete_ratio = test_delete_ratio

    def _delete_severe_se_in_test(self, split_indices):
        train_ri, train_ci, val_ri, val_ci, test_ri, test_ci = split_indices
        test_drug_indices = np.unique(test_ri)
        severe_indices = np.arange(self.binary_data.shape[1])[-68:]

        train_ri_test_drug = train_ri[np.isin(train_ri, test_drug_indices)]
        train_ci_test_drug = train_ci[np.isin(train_ri, test_drug_indices)]

        severe_train_ri = train_ri_test_drug[
            np.isin(train_ci_test_drug, severe_indices)
        ]
        severe_train_ci = train_ci_test_drug[
            np.isin(train_ci_test_drug, severe_indices)
        ]
        self.binary_data[severe_train_ri, severe_train_ci] = 0

    def _randomly_delete_entries(self, split_indices):

        if self.test_delete_ratio == 0:
            return

        train_ri, train_ci, val_ri, val_ci, test_ri, test_ci = split_indices
        test_drug_indices = np.unique(test_ri)

        train_ri_test_drug = train_ri[np.isin(train_ri, test_drug_indices)]
        train_ci_test_drug = train_ci[np.isin(train_ri, test_drug_indices)]

        # deleting probs are weighted according to number of known side effects
        tmp = copy.deepcopy(self.binary_data[train_ri_test_drug, train_ci_test_drug])
        tmp_df = pd.DataFrame({"key": train_ri_test_drug, "target": tmp})
        weights_for_drugs = tmp_df.groupby("key")["target"].sum()
        weights_for_entries = (
            pd.Series(train_ri_test_drug).apply(lambda x: weights_for_drugs[x]).values
        )

        pos_idx = tmp.nonzero()[0]  # index of pos in tmp
        pos_weights = weights_for_entries[pos_idx]
        num_pos = tmp.sum()

        pos_probs = pos_weights / pos_weights.sum()  # type: ignore
        np.random.seed(self.random_seed)
        delete = np.random.choice(
            np.arange(num_pos),
            size=int(num_pos * self.test_delete_ratio),
            p=pos_probs,
            replace=False,
        )
        delete_idx = pos_idx[delete]

        tmp[delete_idx] = 0
        self.binary_data[train_ri_test_drug, train_ci_test_drug] = tmp

    def process(self, split_indices):
        self._delete_severe_se_in_test(split_indices)
        self._randomly_delete_entries(split_indices)
        return self.binary_data


def get_loader(binary_data, split_indices) -> tuple[DataLoader, ...]:
    train_loader = DataLoader(
        ADRDataset(binary_data, idx_i=split_indices[0], idx_c=split_indices[1]),
        batch_size=512,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    val_loader = DataLoader(
        ADRDataset(binary_data, idx_i=split_indices[2], idx_c=split_indices[3]),
        batch_size=2048,
        num_workers=2,
        pin_memory=False,
    )
    if len(split_indices) == 4:
        return train_loader, val_loader
    else:
        test_loader = DataLoader(
            ADRDataset(binary_data, idx_i=split_indices[4], idx_c=split_indices[5]),
            batch_size=2048,
            num_workers=2,
            pin_memory=False,
        )
        return train_loader, val_loader, test_loader


class ADRDataset(Dataset):
    def __init__(self, train, idx_i, idx_c):
        """
        data: np.array of input data (n_drug, n_adr)
        """
        self.out = []  # (drug_idx, adr_idx, target)
        for i, c in zip(idx_i, idx_c):
            t = train[i, c]
            self.out.append((i, c, t))

    def __len__(self):
        return len(self.out)

    def __getitem__(self, idx):
        i, c, t = self.out[idx]
        return i, c, t
