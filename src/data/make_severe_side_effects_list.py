import pickle

import pandas as pd


processed_sider = pd.read_csv("./data/processed_sider.csv", header=0, index_col=0)
adr_cols = [col for col in processed_sider.columns.values if col not in ["flat_id", "name", "SMILES_string"]]  # from _old_02_make_sider_dataset.ipynb

# 日本語名: Preferd Term, from 重篤副作用疾患別対応マニュアル
hihu_adr_dic = {
    "スティーヴンス・ジョンソン症候群": ["Stevens-Johnson syndrome"],
    "中毒性表皮壊死融解症": ["Toxic epidermal necrolysis"],
    "薬剤性過敏症症候群": ["Drug rash with eosinophilia and systemic symptoms"],
    "急性汎発性発疹性膿疱症": ["Acute generalised exanthematous pustulosis"],
    "薬剤による接触皮膚炎": ["Dermatitis contact"],
    "多形紅斑": ["Erythema multiforme"],
}


kanzo_adr_dic = {"薬物性肝障害": ["Drug-induced liver injury"]}


jinzo_adr_dic = {
    "急性腎障害": ["Acute kidney injury"],
    "間質性腎炎": ["Tubulointerstitial nephritis"],
    "ネフローゼ症候群": ["Nephrotic syndrome"],
    "血管炎による腎障害": ["Vasculitis"],
    "腫瘍崩壊症候群": ["Tumour lysis syndrome"],
    "腎性尿崩症": [
        "Nephrogenic diabetes insipidus",
        "Congenital nephrogenic diabetes insipidus",
    ],
    "低カリウム血症": ["Hypokalaemia"],
}


ketueki_adr_dic = {
    "再生不良性貧血": ["Aplastic anaemia", "Congenital aplastic anaemia", "Pancytopenia"],
    "薬剤性貧血": [col for col in adr_cols if "anaemia" in col.lower()],
    "出血傾向": [
        col
        for col in adr_cols
        if "haemorrhage" in col.lower() or "haemorrhagic" in col.lower()
    ],
    "無顆粒球症": [
        "Granulocytopenia",
        "Granulocytopenia neonatal",
        "Granulocyte count decreased",
        "Agranulocytosis",
        "Infantile genetic agranulocytosis",
        "Neutropenia",
        "Neutrophil count decreased",
        "CSF neutrophil count decreased",
        "Neutropenic infection",
        "Neutropenic colitis",
        "Neutropenic sepsis",
        "Autoimmune neutropenia",
        "Neutropenia neonatal",
        "Idiopathic neutropenia",
        "Febrile neutropenia",
    ],
    "血小板減少症": ["Thrombocytopenia", "Platelet count decreased"],
    "血栓症": [col for col in adr_cols if "thrombosis" in col.lower()],
    "播種性血管内凝固": [
        "Disseminated intravascular coagulation",
        "Disseminated intravascular coagulation in newborn",
    ],
    "血栓性血小板減少性紫斑病": ["Thrombotic thrombocytopenic purpura"],
    "ヘパリン起因性血小板減少症": ["Heparin-induced thrombocytopenia"],
}


kokyuki_adr_dic = {
    "間質性肺炎": ["Interstitial lung disease"],
    "急性肺損傷・急性呼吸窮迫症候群（急性呼吸促迫症候群)": [
        "Lung injury",
        "Acute respiratory distress syndrome",
        "Transfusion-related acute lung injury",
    ],
    "肺水腫": [
        "Pulmonary oedema",
        "Acute pulmonary oedema",
        "Non-cardiogenic pulmonary oedema",
        "Pulmonary oedema neonatal",
        "Reexpansion pulmonary oedema",
        "Pulmonary oedema post fume inhalation",
    ],
    "急性好酸球性肺炎": ["Eosinophilic pneumonia acute"],
    "肺胞出血": ["Pulmonary alveolar haemorrhage"],
    "胸膜炎、胸水貯留": [
        "Pleurisy",
        "Pleurisy viral",
        "Tuberculous pleurisy",
        "Malignant pleural effusion",
        "Pleural fibrosis",
    ],
}

syokaki_adr_dic = {
    "麻痺性イレウス": ["Ileus paralytic"],
    "消化性潰瘍": [
        "Peptic ulcer reactivated",
        "Peptic ulcer haemorrhage",
        "Peptic ulcer",
        "Peptic ulcer perforation",
        "Peptic ulcer, obstructive",
        "Peptic ulcer perforation, obstructive",
        "Stress ulcer",
        "Gastrointestinal ulcer",
        "Diverticular perforation",
        "Gastrointestinal ulcer haemorrhage",
        "Anastomotic ulcer haemorrhage",
        "Gastrointestinal erosion",
        "Gastrointestinal perforation",
        "Gastrointestinal ulcer perforation",
        "Anastomotic ulcer perforation",
        "Anastomotic ulcer",
        "Anastomotic ulcer, obstructive",
    ],
    "偽膜性大腸炎": ["Pseudomembranous colitis"],
    "急性膵炎": ["Pancreatitis acute"],
    "重度の下痢": [
        "Viral diarrhoea",
        "Diarrhoea",
        "Diarrhoea infectious",
        "Diarrhoea haemorrhagic",
        "Bacterial diarrhoea",
        "Post procedural diarrhoea",
        "Diarrhoea neonatal",
        "Diarrhoea infectious neonatal",
    ],
}


sinzo_adr_dic = {
    "心室頻拍": ["Ventricular tachycardia", "Ventricular tachyarrhythmia"],
    "うっ血性心不全": ["Cardiac failure congestive"],
}


sinkei_adr_dic = {
    "薬剤性パーキンソニズム": ["Parkinsonism"],
    "白質脳症": ["Leukoencephalopathy"],
    "横紋筋融解症": ["Rhabdomyolysis"],
    "末梢神経障害": ["Neuropathy peripheral"],
    "ギラン・バレー症候群": ["Guillain-Barre syndrome"],
    "ジスキネジア": ["Dyskinesia", "Tardive dyskinesia"],
    "痙攣・てんかん": [
        "Alcoholic seizure",
        "Epilepsy",
        "Epileptic aura",
        "Status epilepticus",
        "Automatism epileptic",
        "Post-traumatic epilepsy",
        "Clonic convulsion",
        "Tonic convulsion",
        "Convulsions local",
        "Acquired epileptic aphasia",
        "Anticonvulsant drug level",
        "Anticonvulsant drug level abnormal",
        "Anticonvulsant drug level decreased",
        "Anticonvulsant drug level therapeutic",
        "Anticonvulsant drug level below therapeutic",
        "Anticonvulsant drug level above therapeutic",
        "Anticonvulsant drug level increased",
        "Convulsion in childhood",
        "Petit mal epilepsy",
        "Convulsion neonatal",
        "Epilepsy congenital",
        "Frontal lobe epilepsy",
        "Temporal lobe epilepsy",
        "Foetal anticonvulsant syndrome",
        "Grand mal convulsion",
        "Hypoglycaemic seizure",
        "Febrile convulsion",
        "Atypical benign partial epilepsy",
        "Generalised non-convulsive epilepsy",
        "Convulsion",
        "Convulsion prophylaxis",
        "Convulsive threshold lowered",
    ],
    "運動失調": ["Ataxia", "Cerebellar ataxia", "Vestibular ataxia", "Cerebral ataxia"],
    "頭痛": [
        "SUNCT syndrome",
        "Eagles syndrome",
        "Post-traumatic headache",
        "Temporomandibular joint syndrome",
        "Ophthalmoplegic migraine",
        "Tension headache",
        "Cluster headache",
        "Cervicogenic headache",
        "Vascular headache",
        "Post lumbar puncture syndrome",
        "Procedural headache",
        "Headache",
        "Postictal headache",
        "Sinus headache",
        "Hemicephalalgia",
        "Chronic paroxysmal hemicrania",
        "Drug withdrawal headache",
        "Exertional headache",
    ],
    "急性散在性脳脊髄炎": ["Acute disseminated encephalomyelitis"],
    "無菌性髄膜炎": ["Meningitis aseptic", "Meningitis noninfective"],
    "小児の急性脳症": ["Encephalopathy neonatal", "Encephalopathy", "Delirium"],
}


ranso_adr_dic = {"卵巣過剰刺激症候群": ["Ovarian hyperstimulation syndrome"]}


seisin_adr_dic = {
    "悪性症候群": ["Neuroleptic malignant syndrome"],
    "薬剤惹起性うつ病": [
        "Depression",
        "Depression suicidal",
        "Agitated depression",
        "Postpartum depression",
        "Depression postoperative",
        "Major depression",
        "Menopausal depression",
    ],
    "アカシジア": ["Akathisia"],
    "セロトニン症候群": ["Serotonin syndrome"],
    "新生児薬物離脱症候群": ["Drug withdrawal syndrome neonatal"],
}


taisya_adr_dic = {
    "偽アルドステロン症": ["Pseudoaldosteronism"],
    "甲状腺中毒症": [
        "Hyperthyroidism",
        "Neonatal thyrotoxicosis",
        "Endocrine ophthalmopathy",
    ],
    "甲状腺機能低下症": [
        "Hypothyroidism",
        "Primary hypothyroidism",
        "Tertiary hypothyroidism",
        "Post procedural hypothyroidism",
        "Congenital hypothyroidism",
        "Secondary hypothyroidism",
    ],
    "高血糖": [
        "Hyperglycaemia",
        "Hyperglycaemic unconsciousness",
        "Hyperglycaemic hyperosmolar nonketotic syndrome",
        "Hyperglycaemic seizure",
        "Diabetic ketoacidotic hyperglycaemic coma",
    ],
    "低血糖": [
        "Hypoglycaemia",
        "Hypoglycaemia neonatal",
        "Shock hypoglycaemic",
        "Hypoglycaemic coma",
        "Hypoglycaemic unconsciousness",
        "Hypoglycaemic encephalopathy",
        "Hypoglycaemic seizure",
        "Hypoglycaemia unawareness",
        "Pseudohypoglycaemia",
        "Hyperinsulinaemic hypoglycaemia",
        "Postprandial hypoglycaemia",
    ],
}


kabinsyo_adr_dic = {
    "アナフィラキシー": [
        "Anaphylactic shock",
        "Anaphylactic reaction",
        "Anaphylactoid shock",
        "Anaphylactoid reaction",
    ],
    "血管性浮腫": ["Angioedema", "Small bowel angioedema", "Hereditary angioedema"],
}


kouku_adr_dic = {
    "薬物性口内炎": [
        "Stomatitis",
        "Aphthous stomatitis",
        "Bovine pustular stomatitis virus infection",
        "Stomatitis necrotising",
        "Necrotising ulcerative gingivostomatitis",
        "Stomatitis haemorrhagic",
        "Contact stomatitis",
        "Stomatitis radiation",
    ]
}


hone_adr_dic = {
    "骨粗鬆症": [
        "Osteoporosis",
        "Osteoporotic fracture",
        "Senile osteoporosis",
        "Osteoporosis postmenopausal",
        "Osteoporosis circumscripta cranii",
        "Osteoporosis prophylaxis",
        "Osteoporosis-pseudoglioma syndrome",
        "Post-traumatic osteoporosis",
        "Chondroporosis",
    ],
    "特発性大腿骨頭壊症": ["Osteonecrosis"],
}


hinyou_adr_dic = {
    "尿閉・排尿困難": [
        "Urinary retention",
        "Urinary retention postoperative",
        "Dysuria",
        "Strangury",
        "Psychogenic dysuria",
    ],
    "出血性膀胱炎": [
        "Cystitis haemorrhagic",
        "Adenoviral haemorrhagic cystitis",
        "Viral haemorrhagic cystitis",
    ],
}


me_adr_dic = {
    "緑内障": [
        "Glaucoma",
        "Uveitis-glaucoma-hyphaema syndrome",
        "Open angle glaucoma",
        "Glaucoma traumatic",
        "Borderline glaucoma",
        "Pigmentary glaucoma",
        "Phacolytic glaucoma",
        "Normal tension glaucoma",
        "Diabetic glaucoma",
        "Developmental glaucoma",
        "Angle closure glaucoma",
        "Glaucoma surgery",
        "Glaucomatous optic disc atrophy",
        "Glaucomatocyclitic crises",
        "Glaucoma drug therapy",
    ],
    "角膜混濁": ["Corneal opacity"],
}


mimi_adr_dic = {
    "難聴": [
        "Deafness transitory",
        "Deafness permanent",
        "Deafness traumatic",
        "Deafness neurosensory",
        "Mixed deafness",
        "Deafness occupational",
        "Deafness congenital",
        "Conductive deafness",
        "Sudden hearing loss",
        "Deafness",
        "Deafness unilateral",
        "Deafness bilateral",
        "Presbyacusis",
    ]
}


kuti_adr_dic = {
    "薬物性味覚障害": [
        "Dysgeusia",
        "Hypergeusia",
        "Gustometry abnormal",
        "Hypogeusia",
        "Ageusia",
    ]
}


gan_adr_dic = {"手足症候群": ["Palmar-plantar erythrodysaesthesia syndrome"]}


dics = [
    hihu_adr_dic,
    kanzo_adr_dic,
    jinzo_adr_dic,
    ketueki_adr_dic,
    kokyuki_adr_dic,
    syokaki_adr_dic,
    sinzo_adr_dic,
    sinkei_adr_dic,
    ranso_adr_dic,
    seisin_adr_dic,
    taisya_adr_dic,
    kabinsyo_adr_dic,
    kouku_adr_dic,
    hone_adr_dic,
    hinyou_adr_dic,
    me_adr_dic,
    mimi_adr_dic,
    kuti_adr_dic,
    gan_adr_dic,
]

pickle.dump(dics, open("./data/severe_adr_dics.pkl", "wb"), -1)