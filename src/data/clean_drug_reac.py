import os
import re
from glob import glob
from itertools import combinations
import pickle

import pandas as pd
from tqdm import tqdm
import numpy as np


def is_valid_id(_id):
    """
    check if id is valid
    """
    return _id.replace(".","").isdigit()


def process_drug(df):
    _df = df.rename(columns={"ISR":"primaryid"})
    _df.columns = _df.columns.str.lower()

    # only include primary suspect
    _df = _df[_df["role_cod"] == "PS"]

    _df = _df[["primaryid","drug_seq", "drugname"]]
    _df = _df[[is_valid_id(str(i)) for i in _df.primaryid]]
    drugname_cleaner = lambda x: re.sub(r'\s+',r' ',str(x)).strip().lower()
    _df.drugname = [drugname_cleaner(name) for name in _df.drugname]
    return _df.astype(
        {"primaryid":int, "drug_seq":int}
    ).drop_duplicates(["primaryid", "drugname"])


def process_reac(df):
    _df = df.rename(columns={"ISR":"primaryid"})
    _df.columns = _df.columns.str.lower()
    _df = _df[["primaryid","pt", "year", "quarter"]]
    _df = _df[[is_valid_id(str(i)) for i in _df.primaryid]]
    _df.pt = _df.pt.str.lower()
    return _df.astype({"primaryid":int}).drop_duplicates(["primaryid", "pt"])


def concat_drug_files(drug_files=None):
    all_df = pd.DataFrame()
    for file in tqdm(drug_files):
        print(f"Procesing {file}")
        df = pd.read_csv(file, sep="$", index_col=False, low_memory=False)
        all_df = all_df.append(process_drug(df))
    return all_df


def concat_reac_files(drug_files=None):
    all_df = pd.DataFrame()
    for file in tqdm(drug_files):
        print(f"Procesing {file}")
        df = pd.read_csv(file, sep="$", index_col=False, low_memory=False)
        df["year"] = file[-8:-6]
        df["quarter"] = file[-5]
        all_df = all_df.append(process_reac(df))
    return all_df


if __name__ == "__main__":
    # get list of raw files
    drug_files = sorted(glob("./data/drug/*"), key=lambda x: x.upper())
    reac_files = sorted(glob("./data/reac/*"), key=lambda x: x.upper())

    drug = concat_drug_files(drug_files)
    reac = concat_reac_files(reac_files)

    all_data = pd.merge(drug, reac, on="primaryid")
    all_data["year"] = all_data["year"].astype(int)
    all_data["quarter"] = all_data["quarter"].astype(int)

    pickle.dump(all_data.drop_duplicates(), open("./data/faers_19q2_data_dup.pkl", "wb"), -1)

