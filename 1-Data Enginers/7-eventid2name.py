import pandas as pd
import numpy as np
import ast

pd.set_option("display.width",1000)
pd.set_option("display.max_columns",500)
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.float_format",lambda x: "%.3f" % x)

df = pd.read_csv(r"Data/eventid2name.csv")
df.head()
df.shape
df.isnull().sum().sort_values(ascending=False)
df.nunique()
df.duplicated().sum()

# EVETN LABEL İLE SUBEVET LABEL ARASI ANLAMLI BİR EŞLEŞME VAR MI
df.groupby("event")["event_label"].nunique().max()
df.groupby("subevent")["subevent_label"].nunique().max()

df.to_csv("event_id_2_name.csv", index=False)

def eventid2_name_pipline(dataframe):
    dataframe.to_csv("event_id_2_namee.csv", index=False)
    return dataframe

eventid2_name_pipline(df)
