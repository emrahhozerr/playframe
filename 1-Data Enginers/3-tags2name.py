import pandas as pd
import numpy as np

pd.set_option("display.width",1000)
pd.set_option("display.max_columns",500)
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.float_format",lambda x: "%.3f" % x)

df = pd.read_csv(r"Data/tags2name.csv")

df.head()
df.shape
df.isnull().sum()
df.dtypes
df.nunique()
df.duplicated().sum()

# df dışarı aktar
df.to_csv("tags_2_name.csv",index=False)

def tag2names_pipline(dataframe):
    dataframe.to_csv("tags_2_namee.csv", index=False)
    return dataframe
tag2names_pipline(df)
