import pickle
import math
from glob import glob
import re
from collections import defaultdict
from collections import Counter
from functools import wraps, lru_cache
import itertools
from copy import deepcopy
import itertools

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def prepare_file(filename):
    data = pd.read_csv(filename, sep="$",index_col=False, low_memory=False, error_bad_lines=False, encoding="utf-8")
    data = data.rename(columns={"ISR":"primaryid","FDA_DT":'fda_dt'}).fillna("")
    data.columns = data.columns.str.lower()

    def is_valid_id(_id):
        return _id.replace(".","").isdigit()

    data = data[[is_valid_id(str(i)) for i in data.primaryid]]
    data = data.applymap(lambda x: re.sub(r'\s+',r' ',str(x)).strip().lower())

    return data.astype({"drugname":str})


# for each drug file
# if product name in cols (thus newer ver. of fares),
# return set of drugname-product name pair
# else return set of drugnames
class DrugnameGetor:

    def __init__(self, filename):
        self.df = prepare_file(filename)

    def check_prodai(self):
        return "prod_ai" in set(self.df.columns)

    def get_drugname(self):
        if self.check_prodai():
            return {"{}_{}".format(name,pro_name)
                    for name, pro_name in zip(self.df.drugname, self.df.prod_ai)}
        else:
            return set(self.df.drugname)

    def main(self):
        return self.get_drugname()


# combine all drug files to get all names
def makenamefolder(drug_filenames):
    drugnames = set()
    for filename in drug_filenames:
        print(filename)
        names = DrugnameGetor(filename).main()
        for name in names:
            drugnames.add(name)
    return drugnames


class LinkingKeggIdDrugName:

    def __init__(self, data, kegg_name_id):
        self.drug_name_set = data
        self.kegg_name_id = kegg_name_id

    def split_drugnames(self, set_drug_name, fragment, how):

        if how=="head_only":

            freq = re.compile(r'tab|oral|blinded|teva|teva-|teva -|novo-')
            drop_freq = lambda x: re.sub(freq, "", x).strip()
            split_fragments = lambda x: [re.split(fragment,drop_freq(x))[0]]
        elif how=="all":
            split_fragments = lambda x: re.split(fragment,x)

        else:
            print("how_split? head_only, all")

        drugs = set_drug_name
        data_fragment = defaultdict(set)

        for drug_names in drugs:
            for drug_name in split_fragments(drug_names):
                if re.search(r'[a-z]', drug_name) and len(drug_name.strip())>1:
                    data_fragment[drug_name.strip()].add(drug_names)
        return data_fragment

    def link_drugname_to_kegg(self):

        dic_id_drugname = defaultdict(set)
        nameid = self.kegg_name_id
        drug_fragment = self.split_drugnames(
            self.drug_name_set, re.compile(r'[^a-z\s\-]'),"all")

        common = set(nameid.keys()) & set(drug_fragment.keys())

        for fragment in tqdm(common):
            drug_names = drug_fragment[fragment]
            ids = nameid[fragment]
            for _id, drug_name in itertools.product(ids, drug_names):
                dic_id_drugname[_id].add(drug_name)

        return dic_id_drugname


    def link_nokegg_to_kegg(self):

        id_name = self.link_drugname_to_kegg()
        id_name2 = deepcopy(id_name)

        no_kegg_name = self.drug_name_set - {ii for i in id_name.values() for ii in i}

        drug_fragment = self.split_drugnames(
            no_kegg_name, re.compile(r'[^a-z\-]'), "head_only")

        for _id, names in tqdm(id_name.items()):
            model_dic=self.split_drugnames(
                names,re.compile(r'[^a-z\-\s]'),"head_only")
            commons = model_dic.keys() & drug_fragment
            for common in commons:
                for drug in drug_fragment[common]:
                    id_name2[_id].add(drug)

        return id_name2


    def get_out_of_kegg(self):
        _, name_id2 = self.link_nokegg_to_kegg()
        return {name for name in self.drug_name_set if not name in name_id2.keys()}


    def main(self):
        idname = self.link_nokegg_to_kegg()
        dic = defaultdict(set)

        for _id,names in idname.items():
            for name in names:
                dic[_id].add(re.split(r'[A-Z]',name)[0])

        return dic


if __name__ == "__main__":
    drug = sorted(glob("./data/drug/*"), key=lambda x: str(x).upper())
    kegg = pickle.load(open("./data/all_keggid.pkl", "rb"))

    dic = LinkingKeggIdDrugName(makenamefolder(drug),kegg[1]).main()
    pickle.dump(dic, open("./data/kegg_faers_proai.pkl","wb"),-1)


