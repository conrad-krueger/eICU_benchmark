import pandas as pd
import sys
import numpy as np
df = pd.read_csv(sys.argv[1])

x = df[np.logical_and(df["nursingchartcelltypevallabel"] == "Glasgow coma score", df["nursingchartcelltypevalname"] == "Eyes")]["nursingchartvalue"].unique()

print("n = ",len(x))
print(x)
