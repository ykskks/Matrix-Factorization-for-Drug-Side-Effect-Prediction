import time
import pickle

import requests
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolStandardize
from tqdm import tqdm

tqdm.pandas()


def get_text(did):
    response = requests.get('http://rest.kegg.jp/get/'+ str(did) + '/mol')
    if response.status_code != 200:
        return np.nan
    else:
        return response.text


def clean_text(text):
    if pd.isnull(text):
        return np.nan

    text = "\n".join(
        line for line in text.split("\n")
        if not ('M  STY' == line[:6] and ('SRU' in line or 'MUL' in line))
    )
    return text


def get_struct(did):
    text = get_text(did)
    text_cleaned =  clean_text(text)

    # courtesy
    time.sleep(1.0)
    if pd.isnull(text_cleaned):
        return np.nan

    mol = Chem.MolFromMolBlock(text_cleaned)

    if mol is not None:
        return mol
    else:
        return np.nan


def process_raw_structure(df):
    _df = df.dropna(axis=0, how="any").reset_index(drop=True)

    # remove inorganics
    _df["smiles"] = _df["mol"].apply(lambda mol: MolToSmiles(mol))
    _df = _df[
        _df["smiles"].str.contains("C") | _df["smiles"].str.contains("c")
    ]
    _df.reset_index(drop=True, inplace=True)

    # remove salts & mixed
    normalizer = MolStandardize.normalize.Normalizer()
    lfc = MolStandardize.fragment.LargestFragmentChooser()
    _df["mols_norm"] = _df["mol"].apply(normalizer.normalize)
    _df["mols_lf"] = _df["mols_norm"].apply(lfc.choose)

    # nuetralize charges
    uc = MolStandardize.charge.Uncharger()
    _df["mols_unc"] = _df["mols_lf"].apply(uc.uncharge)

    # preprocesed smiles
    # salts are now converted to the same smiles
    _df["smiles_final"] = _df["mols_unc"].apply(lambda mol: MolToSmiles(mol))

    return _df



if __name__ == "__main__":
    # get structure data
    df = pd.DataFrame()
    df["id"] = [f"D{str(i).zfill(5)}" for i in range(1, 12000)]
    df["mol"] = df["id"].progress_apply(get_struct)

    pickle.dump(df, open("./data/kegg_drug_mol.pkl", "wb"), -1)

    df = process_raw_structure(df)

    unique_smiles = df["smiles_final"].unique()
    ids = df["id"].values
    smiles_finals = df["smiles_final"].values

    equivalent_drug_set_by_id = [
        set(keggid for keggid in ids
        if smiles_finals[ids == keggid] == smiles)
        for smiles in tqdm(unique_smiles)
    ]

    pickle.dump(equivalent_drug_set_by_id, open("./data/same_drug_set_by_id.pkl", "wb"), -1)

