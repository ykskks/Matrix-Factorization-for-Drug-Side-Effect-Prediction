import time
from itertools import combinations
import re
from base64 import b64decode

from sklearn.metrics import jaccard_score
import requests
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MolToSmiles
from rdkit.Chem import MolStandardize
from tqdm import tqdm
from mordred import Calculator, descriptors

tqdm.pandas()

def get_text(did):
    response = requests.get("http://rest.kegg.jp/get/" + str(did) + "/mol")
    if response.status_code != 200:
        return np.nan
    else:
        return response.text


def clean_text(text):
    if pd.isnull(text):
        return np.nan

    text = "\n".join(
        line
        for line in text.split("\n")
        if not ("M  STY" == line[:6] and ("SRU" in line or "MUL" in line))
    )
    return text


def get_struct(did):
    text = get_text(did)
    text_cleaned = clean_text(text)

    time.sleep(0.5)
    if pd.isnull(text_cleaned):
        return np.nan

    mol = Chem.MolFromMolBlock(text_cleaned)

    if mol is not None:
        return mol
    else:
        return np.nan

def calculate_fingerprints(mols, mode):
    if mode == "maccs":
        maccs = pd.DataFrame([np.array(AllChem.GetMACCSKeysFingerprint(mol)).tolist() for mol in mols])
        return maccs
    
    if mode == "morgan":
        morgan = pd.DataFrame([np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)).tolist() for mol in mols])
        return morgan
    
    if mode == "morgan_count":
        morgan_count = pd.DataFrame([get_morgan_count(mol).tolist() for mol in mols])
        return morgan_count


def jaccard_similarity(a, b):
    """
    a, b: numpy array, binary
    """
    intersection = ((a + b) == 2).sum()
    union = a.sum() + b.sum() - intersection
    if union == 0:
        return 0
    return intersection / union


def compute_drug_similarity(X, metric):
    if metric == "tanimoto":
        sim_func = DataStructs.TanimotoSimilarity
    if metric == "pearson":
        sim_func = np.corrcoef
    if metric == "jaccard":
        sim_func = jaccard_similarity
        
    S = np.zeros((X.shape[0], X.shape[0]))
    seq = range(X.shape[0])
    for a, b in tqdm(list(combinations(seq, 2))):
        if metric == "tanimoto" or metric == "jaccard":
            X_a = X[a, :]
            X_b = X[b, :]
            score = sim_func(X_a, X_b)

        if metric == "pearson":
            X_a = X[a, :]
            X_b = X[b, :]
            score = sim_func(X_a, X_b)
            score = score[0][1]
        S[a, b] = score
    return S


def get_keggid2pubchem():
    keggid2pubchem = {}
    url = "http://rest.kegg.jp/conv/drug/pubchem"
    text = requests.get(url).text
    lines = text.split("\n")
    for line in lines:
        if not line:
            continue
        pubchem, kegg = line.split("\t")
        keggid2pubchem[kegg.split(":")[1]] = pubchem.split(":")[1]
    return keggid2pubchem

def get_pubchem_fp(pubchem_id):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_id}/property/Fingerprint2D/xml"
    text = requests.get(url).text
    # print(re.search(r'<Fingerprint2D>(.*?)</Fingerprint2D>', text))
    if "Fingerprint2D" not in text:
        return None
    fp = text.split("Fingerprint2D")[-2].replace("</", "").replace(">", "").replace(" ", "")
    return fp

def PCFP_BitVector(pcfp_base64):
    if pcfp_base64 is None:
        return np.zeros(881)
    pcfp_bitstring = "".join(["{:08b}".format(x) for x in b64decode(pcfp_base64)])[32:913]
    return np.array(list(pcfp_bitstring), dtype=int)

def calculate_pubchem_fingerprints(keggids):
    keggid2pubchem = get_keggid2pubchem()
    return np.array([PCFP_BitVector(get_pubchem_fp(keggid2pubchem[i])) for i in keggids])


def get_sim_matrix(path, type_):
    train = pd.read_csv(path, index_col="keggid_new")
    print(train.shape)
    train["mol"] = train.index.to_frame()["keggid_new"].progress_apply(get_struct)
    
    if type_ == "maccs":
        fp = calculate_fingerprints(train["mol"].values, mode=type_)
    elif type_ == "morgan":
        fp = calculate_fingerprints(train["mol"].values, mode=type_)
    elif type_ == "pubchem":
        fp = calculate_pubchem_fingerprints(train.index.to_frame()["keggid_new"].values)
    else:
        raise ValueError("Not valid input.")

    if isinstance(fp, pd.DataFrame):
        fp = fp.values

    print(fp.shape)
    S = compute_drug_similarity(fp, "jaccard")
    S_full = S + S.T
    return S_full


if __name__ == "__main__":
    type_ = "pubchem"
    sim_matrix = get_sim_matrix("./data/crosstab_train.csv", type_)
    np.save("./data/pubchem_sim_matrix.npy", sim_matrix)
    # print(get_keggid2pubchem())
    # fp = get_pubchem_fp(7847606)
    # print( len(PCFP_BitVector(fp)) )
    # print(PCFP_BitVector(fp))