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

    print("[PHENO] Reading Diagnosis Table...")
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
    print("[PHENO] Format listfile...")
    label=label.rename(columns={"patientunitstayid": "stay", "itemoffset":"period_length"})
    label["period_length"] = diag.groupby('patientunitstayid')["itemoffset"].max()/60

    label = label.reset_index()
    label[label_pheno] = np.where(label[label_pheno] >= 1, 1, label[label_pheno])
    label=label.rename(columns={"patientunitstayid": "stay", "itemoffset":"period_length"})
    label["stay"] = label["stay"].apply(lambda x: f"{x}_episode1_timeseries.csv")

    label[label_pheno] = label[label_pheno].astype(int)
    print("[PHENO] Train-Val-Test Split...")
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
    df = pd.DataFrame(columns=["stay", "period_length", "y_true"], dtype=int)
    n = len(os.listdir("./data_mimicformat/phenotyping/train"))
    for i,filename in enumerate(os.listdir("./data_mimicformat/phenotyping/train")):
        print(i/n)
        df1 = pd.read_csv("./data_mimicformat/phenotyping/train/"+filename)
        mx_rlos = df1["itemoffset"].max()

        df_curr = {"stay":[], "period_length":[], "y_true":[]}
        for t in range(5, int(mx_rlos)):
            df_curr["stay"].append(filename)
            df_curr["period_length"].append(t)
            df_curr["y_true"].append(float(mx_rlos)-t)
        
        df = pd.concat([df, pd.DataFrame(df_curr)])
    for filename in os.listdir("./data_mimicformat/phenotyping/test"):
        df1 = pd.read_csv("./data_mimicformat/phenotyping/test/"+filename)
        mx_rlos = df1["itemoffset"].max()

        df_curr = {"stay":[], "period_length":[], "y_true":[]}
        for t in range(5, int(mx_rlos)):
            df_curr["stay"].append(filename)
            df_curr["period_length"].append(t)
            df_curr["y_true"].append(float(mx_rlos)-t)
        
        df = pd.concat([df, pd.DataFrame(df_curr)])
        
    
    #shuffle
    df = df.sample(frac = 1)

    if split:
        tr = set(tr)
        tst = set(tst)
        val = set(val)
        train = df.loc[df["stay"].apply(lambda x: x in tr)]
        test = df.loc[df["stay"].apply(lambda x: x in tst)]
        val = df.loc[df["stay"].apply(lambda x: x in val)]

        # df_id = df.set_index("stay")
        # train = df_id.loc[tr].reset_index()
        # test = df_id.loc[tst].reset_index()
        # val = df_id.loc[val].reset_index()
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

def get_decomp_listfile(pats, pat_ids, tr, tst, val, split):
    if not os.path.exists("./data_mimicformat/decompensation"):
        os.mkdir("./data_mimicformat/decompensation")

    if not os.path.exists("./data_mimicformat/length-of-stay/"):
        os.mkdir("./data_mimicformat/length-of-stay/")

    df = pd.DataFrame(columns=["stay", "period_length", "y_true"], dtype=int)
    df["stay"] = pats["patientunitstayid"]
    df["period_length"] = pats["unitdischargeoffset"]/60
    df["y_true"] = pats["hospitaldischargestatus"].apply(lambda x: int(x != "Alive"))


    df = df.loc[df["stay"].apply(lambda x: os.path.exists(f"./data_temp/{x}_episode1_timeseries.csv"))]
    df["stay"] = pats["patientunitstayid"].apply(lambda x: f"{x}_episode1_timeseries.csv")

    df_id = df.set_index("stay")
    train = df_id.loc[tr].reset_index()
    test = df_id.loc[tst].reset_index()
    val = df_id.loc[val].reset_index()

    with open("./data_mimicformat/decompensation/train_listfile.csv", "w") as f:
        f.write("stay,period_length,y_true,y_remain\n")
        for i in range(1,len(train)):
            row = train.iloc[i]
            print("Train",i/len(train))
            st, pl, y = row['stay'], row['period_length'], row['y_true']
            for i in range(1,int(pl)-10):
                if pl - i <= 24 and y == 1:
                    f.write(f"{st},{pl - i},{y},{i}\n")
                else:
                    f.write(f"{st},{pl - i},0,{i}\n")
    
    train = pd.read_csv("./data_mimicformat/decompensation/train_listfile.csv")
    train1 = train.sample(frac = 1).drop("y_remain",axis=1)
    train1.to_csv("./data_mimicformat/decompensation/train_listfile.csv", index=False)
    train2 = train.sample(frac = 1).drop("y_true",axis=1).rename(columns={"y_remain":"y_true"})
    train2.to_csv("./data_mimicformat/length-of-stay/train_listfile.csv", index=False)


    with open("./data_mimicformat/decompensation/val_listfile.csv", "w") as f:
        f.write("stay,period_length,y_true,y_remain\n")
        for i in range(1,len(val)):
            row = val.iloc[i]
            print(i/len(val))
            st, pl, y = row['stay'], row['period_length'], row['y_true']
            for i in range(1,int(pl)-10):
                if pl - i <= 24 and y == 1:
                    f.write(f"{st},{pl - i},{y},{i}\n")
                else:
                    f.write(f"{st},{pl - i},0,{i}\n")
    
    train = pd.read_csv("./data_mimicformat/decompensation/val_listfile.csv")
    train1 = train.sample(frac = 1).drop("y_remain",axis=1)
    train1.to_csv("./data_mimicformat/decompensation/val_listfile.csv", index=False)
    train2 = train.sample(frac = 1).drop("y_true",axis=1).rename(columns={"y_remain":"y_true"})
    train2.to_csv("./data_mimicformat/length-of-stay/val_listfile.csv", index=False)


    with open("./data_mimicformat/decompensation/test_listfile.csv", "w") as f:
        f.write("stay,period_length,y_true,y_remain\n")
        for i in range(1,len(val)):
            row = val.iloc[i]
            print(i/len(val))
            st, pl, y = row['stay'], row['period_length'], row['y_true']
            for i in range(1,int(pl)-10):
                if pl - i <= 24 and y == 1:
                    f.write(f"{st},{pl - i},{y},{i}\n")
                else:
                    f.write(f"{st},{pl - i},0,{i}\n")
    
    train = pd.read_csv("./data_mimicformat/decompensation/test_listfile.csv")
    train1 = train.sample(frac = 1).drop("y_remain",axis=1)
    train1.to_csv("./data_mimicformat/decompensation/test_listfile.csv", index=False)
    train2 = train.sample(frac = 1).drop("y_true",axis=1).rename(columns={"y_remain":"y_true"})
    train2.to_csv("./data_mimicformat/length-of-stay/test_listfile.csv", index=False)

    #train.to_csv("./data_mimicformat/decompensation/train_listfile.csv")

    
    if split:
        # train, test, _, _ = train_test_split(df, df["y_true"], test_size=0.15, stratify=df["y_true"])
        # train, val, _, _ = train_test_split(train, train["y_true"], test_size=0.176470588, stratify=train["y_true"])
        # val.to_csv("./data_mimicformat/decompensation/val_listfile.csv", index=False)
        # train.to_csv("./data_mimicformat/decompensation/train_listfile.csv", index=False)
        # test.to_csv("./data_mimicformat/decompensation/test_listfile.csv", index=False)
        
        if not os.path.exists("./data_mimicformat/decompensation/test"):
            os.mkdir("./data_mimicformat/decompensation/test")
        shutil.copytree("./data_temp", "./data_mimicformat/decompensation/train")

        if not os.path.exists("./data_mimicformat/length-of-stay/test"):
            os.mkdir("./data_mimicformat/length-of-stay/test")
        shutil.copytree("./data_temp", "./data_mimicformat/length-of-stay/train")

        for name in test["stay"]:
            try:
                os.rename(f"./data_mimicformat/decompensation/train/{name}", f"./data_mimicformat/decompensation/test/{name}")
            except Exception as e:
                print(e)
                continue  
        for name in test["stay"]:
            try:
                os.rename(f"./data_mimicformat/length-of-stay/train/{name}", f"./data_mimicformat/length-of-stay/test/{name}")
            except Exception as e:
                print(e)
                continue  

    return train["stay"], val["stay"], test["stay"]
    
    

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
    parser.add_argument("--reset", action="store_true")  
    args = parser.parse_args()
    
    if args.move:
        move_files("output")
    
    if args.reset:
        shutil.rmtree("data_mimicformat/phenotyping")
    
    pats = pd.read_csv("data/patient.csv")
    pat_ids = set(os.listdir("output"))
    pat_ids = set([int(i) for i in pat_ids if i.isnumeric()])
    
    
    pat_ids = set([int(filename.split("_")[0]) for filename in os.listdir(f"./data_temp/")])

    if args.all or args.pheno:
        tr, val, tst = get_pheno_listfile(pat_ids, args.split)

    if args.all or args.ihm:
        get_mortality_listfile(pats, pat_ids, args.split)
    
    if args.all or args.los:
        get_rlos_listfile(pats, tr, val, tst, args.split)

    if args.all or args.decomp:
        get_decomp_listfile(pats, pat_ids, tr, val, tst, args.split)
        

    #get_pheno_listfile(pat_ids).to_csv("pheno_listfile.csv")

    #print(get_pheno_listfile(pat_ids))

    #print(utils.label_decompensation(df))
    print()

    



main()








