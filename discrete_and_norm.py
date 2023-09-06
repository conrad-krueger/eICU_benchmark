from torchmimic_preprocess import Discretizer, Normalizer
import pandas as pd
import numpy as np
from data_extractor import utils


def drop_before_zero():
    pass

PROPER_ORDER = ["itemoffset","Eyes","FiO2","GCS Total","Heart Rate","Invasive BP Diastolic","Invasive BP Systolic","Motor","O2 Saturation","Respiratory Rate","Temperature (C)","Verbal","glucose","pH","MAP (mmHg)","Capillary Refill","admissionheight","admissionweight"]

def discretize(filename, outfile):
    discretizer = Discretizer(
                timestep=1.0,
                store_masks=True,
                impute_strategy="previous",
                start_time="relative",
            )
    df = pd.read_csv(filename)[PROPER_ORDER]

    #itemoffset,Eyes,GCS Total,Heart Rate,Invasive BP Diastolic,Invasive BP Systolic,Motor,O2 Saturation,Respiratory Rate,Temperature (C),Verbal,glucose,patientunitstayid,MAP (mmHg),pH,FiO2,Capillary Refill,admissionheight,hospitaladmitoffset,admissionweight,hospitaldischargestatus,unitdischargeoffset,unitdischargestatus
    arr = df.to_numpy()
    arr2, header = discretizer.transform(arr)

    DF = pd.DataFrame(arr2)
    
    # save the dataframe as a csv file
    DF.to_csv(outfile)


import os 
directory = "output"
fail = 0
n = len(os.listdir(directory))
for i,filename in enumerate(os.listdir(directory)):
    print(i/n)
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isdir(f):
        print(f)
        #try:
        discretize(f+"/timeseries1.csv",f+"/timeseries_change.csv")
        #except Exception as e:
        #    print("ERR:",e)
        #    fail += 1

print(fail, n, fail/len(os.listdir(directory)))

    

#utils.delete_wo_timeseries("./output")
# #Write all the data into one dataframe
#utils.all_df_into_one_df("./output")