Note: Due to the KEGG api update, the created dataset will slightly differ from the original dataset used in the paper. However, we confirmed that this change did not affect the result and discussion. To reproduce the exact same result, please use the original dataset under ./data.

# Setup
```
pip install -r requirements.txt
```

# Data
Download FAERS REAC & DRUG tables (2004Q1-2019Q2) under ./data/ directory

# Run
```
# preprocess
bash ./src/data/data.sh

# MF
python ./src/models/mf.py

# external test for MF
python ./src/models/mf_external_test.py

# sider for MF
python ./src/models/mf_sider.py
```
