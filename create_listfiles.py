import pandas as pd
import os
import numpy as np
from data_extractor import utils
import json
import argparse
from sklearn.model_selection import train_test_split
import shutil
import os
 

def get_pheno_listfile(pat_ids, split):
    if not os.path.exists("./data_mimicformat/phenotyping"):
        os.mkdir("./data_mimicformat/phenotyping")

    label_pheno = ['Respiratory failure', 'Essential hypertension',
                'Cardiac dysrhythmias', 'Fluid disorders', 'Septicemia',
                'Acute and unspecified renal failure', 'Pneumonia',
                'Acute cerebrovascular disease', 'CHF', 'CKD', 'COPD',
                'Acute myocardial infarction', "Gastrointestinal hem",
                'Shock', 'lipid disorder', 'DM with complications', 'Coronary athe',
                'Pleurisy', 'Other liver diseases', 'lower respiratory',
                'Hypertension with complications', 'Conduction disorders',
                'Complications of surgical', 'upper respiratory',
                'DM without complication']

    diag_ord_col = ["patientunitstayid", "itemoffset", "Respiratory failure", "Fluid disorders",
                    "Septicemia", "Acute and unspecified renal failure", "Pneumonia",
                    "Acute cerebrovascular disease",
                    "Acute myocardial infarction", "Gastrointestinal hem", "Shock", "Pleurisy",
                    "lower respiratory", "Complications of surgical", "upper respiratory",
                    "Hypertension with complications", "Essential hypertension", "CKD", "COPD",
                    "lipid disorder", "Coronary athe", "DM without complication",
                    "Cardiac dysrhythmias",
                    "CHF", "DM with complications", "Other liver diseases", "Conduction disorders"]

    diag_columns = ['patientunitstayid', 'itemoffset','Respiratory failure','Essential hypertension', 'Cardiac dysrhythmias',
                'Fluid disorders', 'Septicemia','Acute and unspecified renal failure', 'Pneumonia',
                'Acute cerebrovascular disease', 'CHF', 'CKD', 'COPD','Acute myocardial infarction', "Gastrointestinal hem",
                'Shock', 'lipid disorder', 'DM with complications', 'Coronary athe','Pleurisy', 'Other liver diseases', 'lower respiratory',
                'Hypertension with complications', 'Conduction disorders','Complications of surgical', 'upper respiratory',
                'DM without complication']


    diag = utils.read_diagnosis_table("data/")
    diag = utils.diag_labels(diag)
    diag.dropna(how='all', subset=label_pheno, inplace=True)
    
    stay_diag = set(diag['patientunitstayid'].unique())
    stay_all = pat_ids
    stay_intersection = stay_all.intersection(stay_diag)

    stay_pheno = list(stay_intersection)

    diag = diag[diag['patientunitstayid'].isin(stay_pheno)]

    diag.rename(index=str, columns={"diagnosisoffset": "itemoffset"}, inplace=True)
    diag = diag[diag_columns]
    label = diag.groupby('patientunitstayid').sum()
    
    #my code
    label=label.rename(columns={"patientunitstayid": "stay", "itemoffset":"period_length"})
    label["period_length"] = diag.groupby('patientunitstayid')["itemoffset"].max()
    print(label)

    label = label.reset_index()
    label[label_pheno] = np.where(label[label_pheno] >= 1, 1, label[label_pheno])
    label=label.rename(columns={"patientunitstayid": "stay", "itemoffset":"period_length"})
    label["stay"] = label["stay"].apply(lambda x: f"{x}_episode1_timeseries.csv")

    label[label_pheno] = label[label_pheno].astype(int)
    print(label["period_length"])
    print(label.columns)
    if split:
        train, test, _, _ = train_test_split(label, label[label_pheno], test_size=0.15)
        train, val, _, _ = train_test_split(train, train[label_pheno], test_size=0.176470588)
        val.to_csv("./data_mimicformat/phenotyping/val_listfile.csv", index=False)
        train.to_csv("./data_mimicformat/phenotyping/train_listfile.csv", index=False)
        test.to_csv("./data_mimicformat/phenotyping/test_listfile.csv", index=False)
        
        if not os.path.exists("./data_mimicformat/phenotyping/test"):
            os.mkdir("./data_mimicformat/phenotyping/test")
        shutil.copytree("./data_temp", "./data_mimicformat/phenotyping/train")

        for name in test["stay"]:
            try:
                os.rename(f"./data_mimicformat/phenotyping/train/{name}", f"./data_mimicformat/phenotyping/test/{name}")
            except Exception as e:
                print(e)
                continue  

    return train["stay"], val["stay"], test["stay"]
    

def get_rlos_listfile(pats, tr, val, tst, split):
    df = pd.DataFrame(columns=["stay", "y_true"], dtype=int)
    for filename in os.listdir("./data_mimicformat/phenotyping/train"):
        df1 = pd.read_csv(filename)
        mx_rlos = utils.filter_rlos_data(df1).max()["RLOS"]
        df.loc[len(df)] = [filename, float(mx_rlos)]
    for filename in os.listdir("./data_mimicformat/phenotyping/test"):
        df1 = pd.read_csv(filename)
        mx_rlos = utils.filter_rlos_data(df1).max()["RLOS"]
        df.loc[len(df)] = [filename, float(mx_rlos)]
    

    if split:
        df_id = df.set_index("stay")
        train = df_id.loc[tr].reset_index()
        test = df_id.loc[tst].reset_index()
        val = df_id.loc[val].reset_index()
        train.to_csv("./data_mimicformat/length-of-stay/train_listfile.csv", index=False)
        test.to_csv("./data_mimicformat/length-of-stay/test_listfile.csv", index=False)
        val.to_csv("./data_mimicformat/length-of-stay/val_listfile.csv", index=False)
        
        if not os.path.exists("./data_mimicformat/length-of-stay/test"):
            os.mkdir("./data_mimicformat/length-of-stay/test")
            shutil.copytree("./data_mimicformat/phenotyping/test/", "./data_mimicformat/length-of-stay/test")
        
        if not os.path.exists("./data_mimicformat/length-of-stay/train"):
            os.mkdir("./data_mimicformat/length-of-stay/train")
            shutil.copytree("./data_mimicformat/phenotyping/train/", "./data_mimicformat/length-of-stay/train")

    return df
    

def get_mortality_listfile(pats, pat_ids, split):
    if not os.path.exists("./data_mimicformat/in-hospital-mortality"):
        os.mkdir("./data_mimicformat/in-hospital-mortality")

    df = pd.DataFrame(columns=["stay", "y_true"], dtype=int)
    df["stay"] = pats["patientunitstayid"]
    df["y_true"] = pats["hospitaldischargestatus"].apply(lambda x: int(x != "Alive"))


    df = df.loc[df["stay"].apply(lambda x: os.path.exists(f"./data_temp/{x}_episode1_timeseries.csv"))]
    df["stay"] = pats["patientunitstayid"].apply(lambda x: f"{x}_episode1_timeseries.csv")
    
    if split:
        train, test, _, _ = train_test_split(df, df["y_true"], test_size=0.15, stratify=df["y_true"])
        train, val, _, _ = train_test_split(train, train["y_true"], test_size=0.176470588, stratify=train["y_true"])
        val.to_csv("./data_mimicformat/in-hospital-mortality/val_listfile.csv", index=False)
        train.to_csv("./data_mimicformat/in-hospital-mortality/train_listfile.csv", index=False)
        test.to_csv("./data_mimicformat/in-hospital-mortality/test_listfile.csv", index=False)
        
        if not os.path.exists("./data_mimicformat/in-hospital-mortality/test"):
            os.mkdir("./data_mimicformat/in-hospital-mortality/test")
        shutil.copytree("./data_temp", "./data_mimicformat/in-hospital-mortality/train")

        for name in test["stay"]:
            try:
                os.rename(f"./data_mimicformat/in-hospital-mortality/train/{name}", f"./data_mimicformat/in-hospital-mortality/test/{name}")
            except Exception as e:
                print(e)
                continue  

    return train["stay"], val["stay"], test["stay"]

def move_files(directory):
    os.mkdir("data_mimicformat")
    os.mkdir("data_temp")
    for filename in os.listdir(directory):
        try:
            f = os.path.join(directory, filename)
            f = os.path.join(f, "timeseries.csv")
            os.rename(f, f"./data_temp/{filename}_episode1_timeseries.csv")
        except Exception as e:
            print(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--all", action="store_true")    
    parser.add_argument("--ihm", action="store_true")  
    parser.add_argument("--los", action="store_true")  
    parser.add_argument("--decomp", action="store_true")  
    parser.add_argument("--pheno", action="store_true")  
    args = parser.parse_args()
    
    if args.move:
        move_files("output")
    
    pats = pd.read_csv("data/patient.csv")
    pat_ids = set(os.listdir("output"))
    pat_ids = set([int(i) for i in pat_ids if i.isnumeric()])
    
    
    pat_ids = set([int(filename.split("_")[0]) for filename in os.listdir(f"./data_temp/")])
    print(pat_ids)

    if args.all or args.pheno:
        tr, val, tst = get_pheno_listfile(pat_ids, args.split)

    if args.all or args.ihm:
        get_mortality_listfile(pats, pat_ids, args.split)
    
    if args.all or args.los:
        get_rlos_listfile(pats, tr, val, tst, args.split)
        

    #get_pheno_listfile(pat_ids).to_csv("pheno_listfile.csv")

    #print(get_pheno_listfile(pat_ids))

    #print(utils.label_decompensation(df))
    print()

    



main()








