import re
from collections import Counter, defaultdict
import pickle
import math

import pandas as pd
from tqdm import tqdm
import numpy as np

tqdm.pandas()


def get_id_from_name(name):
    if name in name2id_kegg and name2id_kegg[name]:
        return list(name2id_kegg[name])[0]
    else:
        for keggid, name_set in id2name_prod.items():
            if name in name_set:
                return keggid
        else:
            return np.nan


def convert_keggid(keggid):
    if keggid not in id2signatureid:
        return np.nan
    return id2signatureid[keggid]


def bind_adr_columns(df):
    adrs_added = []
    adrs_jp = []

    for dic in severe_adr_dics:
        for adr_jp, adr_list in dic.items():
            adr_list = [adr.lower() for adr in adr_list if adr.lower() in df.columns.values]
            df[adr_jp] = df[adr_list].sum(axis=1)
            df.drop(adr_list, axis=1, inplace=True)
    return df


def get_id2signatureid(same_drug_set_by_id):
    id2signatureid = {}
    for se in same_drug_set_by_id:
        signature_id = next(iter(se))
        for element in se:
            id2signatureid[element] = signature_id
    return id2signatureid


def main():
    # only use data leading up to 2015Q3 for train
    data_test = data[
        (data["year"] >= 16) |
        ((data["year"] == 15) & (data["quarter"] == 4))
    ]

    data_test["drugname"] = data_test["drugname"].progress_apply(lambda name: name.strip("."))
    data_test["keggid"] = data_test["drugname"].progress_apply(get_id_from_name)
    data_test["keggid_new"] = data_test["keggid"].progress_apply(convert_keggid)
    data_test = data_test.dropna(how="any", axis=0)

    crosstab_test = pd.crosstab(data_test["keggid_new"], data_test["pt"])
    crosstab_test = bind_adr_columns(crosstab_test)

    crosstab_train = pd.read_csv("./data/crosstab_train.csv", index_col="keggid_new")
    tmp = crosstab_test.copy(deep=True)
    res = tmp.reindex(
        index=crosstab_train.index,
        columns=crosstab_train.columns,
        fill_value=0
    )
    crosstab_test = res.copy(deep=True)
    crosstab_test.to_csv("./data/crosstab_test.csv")


if __name__ == "__main__":
    data = pickle.load(open("./data/faers_19q2_data_dup.pkl","rb"))
    name2id_kegg = pickle.load(open("./data/all_keggid.pkl","rb"))[1]
    id2name_prod = pickle.load(open("./data/kegg_faers_proai.pkl","rb"))
    same_drug_set_by_id = pickle.load(open("./data/same_drug_set_by_id.pkl","rb"))
    id2signatureid = get_id2signatureid(same_drug_set_by_id)
    severe_adr_dics = pickle.load(open("./data/severe_adr_dics.pkl","rb"))

    main()
