import pandas as pd
import numpy as np
import ast

pd.set_option("display.width",1000)
pd.set_option("display.max_columns",500)
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.float_format",lambda x: "%.3f" % x)

df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data\events_England.csv")
df.head()
df.shape
df.isnull().sum().sort_values(ascending=False)
df.nunique()
df.duplicated().sum()

# eski (bozuk) pos kolonlarini once at
df = df.drop(columns=["pos_orig_x", "pos_orig_y", "pos_dest_x", "pos_dest_y"])

# pos_orig_x/y, pos_dest_x/y kolonlarini positions'tan kendimiz dogru sekilde uretiyoruz
pattern = r"\{'y':\s*(-?\d+),\s*'x':\s*(-?\d+)\}"
matches = df["positions"].str.findall(pattern)
df["pos_orig_y"] = matches.apply(lambda m: int(m[0][0]) if len(m) > 0 else np.nan)
df["pos_orig_x"] = matches.apply(lambda m: int(m[0][1]) if len(m) > 0 else np.nan)
df["pos_dest_y"] = matches.apply(lambda m: int(m[1][0]) if len(m) > 1 else np.nan)
df["pos_dest_x"] = matches.apply(lambda m: int(m[1][1]) if len(m) > 1 else np.nan)

# bilgi tekrarı değişkelerde
df.drop(["positions", "tags"], axis=1, inplace=True)

# Tag kısmı ayır değişkenlere aldık bir kullanıcı bir den fazla tagı pozisyonu var
# tag_list ıd olarak değiştir listeden çıkar
def parse_tags(x):
    if isinstance(x, list):
        return x
    if pd.isnull(x):
        return []
    return ast.literal_eval(x)

df["tagsList"] = df["tagsList"].apply(parse_tags)

exploded = df["tagsList"].explode()
tag_dummies = pd.crosstab(exploded.index, exploded)
tag_dummies.columns = [f"tag_{int(c)}" for c in tag_dummies.columns]
tag_dummies = (tag_dummies > 0).astype(int)

df = df.join(tag_dummies)
tag_cols = tag_dummies.columns
df[tag_cols] = df[tag_cols].fillna(0).astype(int)

# artık işimiz olmadığı için çıkardık
df = df.drop(columns=["tagsList"])

df.to_csv("event_england.csv",index=False)






def events_eng_pipline(dataframe):
    import pandas as pd
    import numpy as np
    import ast

    dataframe = dataframe.drop(columns=["pos_orig_x", "pos_orig_y", "pos_dest_x", "pos_dest_y"])

    # pos_orig_x/y, pos_dest_x/y kolonlarini positions'tan kendimiz dogru sekilde uretiyoruz
    pattern = r"\{'y':\s*(-?\d+),\s*'x':\s*(-?\d+)\}"
    matches = dataframe["positions"].str.findall(pattern)
    dataframe["pos_orig_y"] = matches.apply(lambda m: int(m[0][0]) if len(m) > 0 else np.nan)
    dataframe["pos_orig_x"] = matches.apply(lambda m: int(m[0][1]) if len(m) > 0 else np.nan)
    dataframe["pos_dest_y"] = matches.apply(lambda m: int(m[1][0]) if len(m) > 1 else np.nan)
    dataframe["pos_dest_x"] = matches.apply(lambda m: int(m[1][1]) if len(m) > 1 else np.nan)

    # bilgi tekrarı değişkelerde
    dataframe.drop(["positions", "tags"], axis=1, inplace=True)

    # Tag kısmı ayır değişkenlere aldık bir kullanıcı bir den fazla tagı pozisyonu var
    # tag_list ıd olarak değiştir listeden çıkar
    def parse_tags(x):
        if isinstance(x, list):
            return x
        if pd.isnull(x):
            return []
        return ast.literal_eval(x)

    dataframe["tagsList"] = dataframe["tagsList"].apply(parse_tags)

    exploded = dataframe["tagsList"].explode()
    tag_dummies = pd.crosstab(exploded.index, exploded)
    tag_dummies.columns = [f"tag_{int(c)}" for c in tag_dummies.columns]
    tag_dummies = (tag_dummies > 0).astype(int)

    dataframe = dataframe.join(tag_dummies)
    tag_cols = tag_dummies.columns
    dataframe[tag_cols] = dataframe[tag_cols].fillna(0).astype(int)

    # artık işimiz olmadığı için çıkardık
    dataframe = dataframe.drop(columns=["tagsList"])

    dataframe.to_csv("event_englandd.csv", index=False)
    return dataframe

events_eng_pipline(df)


