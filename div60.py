import os
n=len(os.listdir("data_temp"))
import pandas as pd
for i,file in enumerate(os.listdir("data_temp")):
    df = pd.read_csv(os.path.join("data_temp",file)).drop("Unnamed: 0", axis=1)
    #df["itemoffset"] = df["itemoffset"]/60
    file_new = "data_temp/"+file
    os.remove(os.path.join("data_temp",file))

    df.to_csv(file_new, index=False)
    print(i/n)
