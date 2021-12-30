import urllib.request
from collections import defaultdict
import pickle

import pandas as pd


def make_drugcode(url, col_names):
    """
    make df with drugid and name
    """
    df = pd.DataFrame()
    drug_list = list()

    with urllib.request.urlopen(url) as f:
        ls = f.read().decode('utf-8').split('\n')
    for l in ls:
        _l = l.split('\t')
        if len(_l) == 2:
            if ';' in _l[1]:
                drugs = _l[1].split(';')
            else:
                drugs = [_l[1]]
            drug_list.append([_l[0], drugs])
    for d in drug_list:
        for drug in d[1]:
            df = df.append(pd.DataFrame([[d[0], drug]]))

    df = df.reset_index(drop=True)
    df[1] = df[1].str.rstrip('(JP17/JAN/USAN/INN)')
    df[1] = df[1].str.rstrip('\(T')
    df[1] = df[1].str.rstrip('\(JP17/NF')
    df[1] = df[1].str.lower().str.strip()
    df.columns = col_names

    return df


def make_linkcode(url, col_names):
    """
    make df of different ids to link them
    """
    df = pd.DataFrame()
    with urllib.request.urlopen(url) as f:
        ls = f.read().decode('utf-8').split('\n')
    for l in ls:
        _l = l.split('\t')
        if len(_l) == 2:
            df = df.append([[_l[0], _l[1]]])
    df.reset_index(drop=True)
    df.columns = col_names

    return df


def make_dict_0(df):
    """
    make dict: drugname -> keggid
    """
    kegg_dict = defaultdict(set)

    key_list = list()
    for key in df['keggid'].str.strip():
        if key not in key_list:
            key_list.append(key)
            _df = df[df['keggid'] == key].reset_index(drop = True)
            kegg_dict[key] = {_df['name'][0]}
            if len(_df) != 1:
                    kegg_dict[key] = {i for i in _df['name']}

    return kegg_dict


def make_dict_1(df):
    """
    nake dict: keggid -> drugname
    """
    kegg_dict = defaultdict(set)
    for i in range(len(df)):
        key = df.iat[i, 1]
        value = df.iat[i, 0].strip()
        kegg_dict[key] = {value}

    return kegg_dict


if __name__ == "__main__":
    kegg_drug = make_drugcode('http://rest.kegg.jp/list/drug', ["keggid", "name"])
    kegg_drug['keggid'] = kegg_drug['keggid'].str.replace('dr:', '').str.strip()

    kegg_drug_tn = make_drugcode('http://rest.kegg.jp/list/ndc', ["another_id", "name"])

    tn_id = make_linkcode('http://rest.kegg.jp/link/ndc/drug', ["keggid", "another_id"])
    tn_id['keggid'] = tn_id['keggid'].str.replace('dr:', '').str.strip()

    # link another_id to keggid
    tn_marged = pd.merge(kegg_drug_tn, tn_id, on='another_id').drop('another_id', axis = 1).drop_duplicates().reset_index(drop = True)

    # concat two mappings: now both using kegg id
    kegg_all = kegg_drug.append(tn_marged, sort=True).drop_duplicates().sort_values('keggid').reset_index(drop = True)

    dict_0 = make_dict_0(kegg_all)
    dict_1 = make_dict_1(kegg_all)

    pickle.dump((dict_0, dict_1, kegg_all), open('./data/all_keggid.pkl','wb'), -1)

