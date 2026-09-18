import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns",500)
pd.set_option("display.width",1000)
pd.set_option("expand_frame_repr",False)
pd.set_option("float_format",lambda x: "%.3f" % x)

eng_df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Feature_Engineer\Data\event_england_merged.csv",low_memory=False)
tags_df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\EDA\Data\event_tags_long.csv")

eng_df.head()
eng_df.isnull().sum().sort_values(ascending=False).head(20)

# players 0 içerdiği taç vb durumlar bizim için önemli değil bu yüzden çıkarıyoruz
eng_players_df = eng_df[eng_df['playerId'] != 0].copy()
eng_players_df.shape
eng_players_df.isnull().sum().sort_values(ascending=False).head(20)

#  match_gameweek tipini hafta olarak kayıt et
eng_players_df["match_gameweek"] = eng_players_df["match_gameweek"].astype("int64")
eng_players_df["match_gameweek"].dtypes

# Karışıklığı önemle için match_winner id ekle
eng_players_df.rename(columns={"match_winner":"match_winner_id"},inplace=True)
eng_players_df["match_winner_id"]

# Eşisiz değer sayısı 2 nin altında olanları columsları sil
unique_value_tow = [col for col in eng_players_df.columns if eng_players_df[col].nunique() < 2]
eng_players_df.drop(unique_value_tow,axis=1,inplace=True)
eng_players_df.isnull().sum().sort_values(ascending=False).head(10)

# Tarih değişkeni tip düzelme
eng_players_df["match_dateutc"] = pd.to_datetime(eng_players_df["match_dateutc"])
eng_players_df["match_dateutc"].dtypes


# match_winner_id hata düzeltmesi
# 2 maçta beraberlik olmadığı halde match_winner_id=0 yazılmış,
# skor karşılaştırmasıyla doğru winner'ı hesaplıyoruz
def compute_winner(row):
    s1, s2 = row["match_team1.score"], row["match_team2.score"]
    if s1 == s2:
        return 0
    elif s1 > s2:
        return row["match_team1.teamId"]
    else:
        return row["match_team2.teamId"]

eng_players_df["match_winner_id_fixed"] = eng_players_df.apply(compute_winner, axis=1)
# kontrol: kaç satır düzeltildi
(eng_players_df["match_winner_id"] != eng_players_df["match_winner_id_fixed"]).sum()

eng_players_df.isnull().sum().sort_values(ascending=False).head(10)
eng_players_df.dtypes

# unkow değerleri nan çevirdik
unknow_col = [col for col in eng_players_df.columns if (eng_players_df[col].astype(str) == "unknow").any()]
eng_players_df["player_currentTeamId"] = eng_players_df["player_currentTeamId"].replace("unknow", np.nan)
eng_players_df["player_foot"] = eng_players_df["player_foot"].replace("unknow", np.nan)

eng_players_df.to_csv("eng_players_df.csv",index=False)

def event_eng_data_preprocessing(dataframe,csv=False):
    import pandas as pd
    import numpy as np

    # players 0 içerdiği taç vb durumlar bizim için önemli değil bu yüzden çıkarıyoruz
    eng_players_df = dataframe[dataframe['playerId'] != 0].copy()

    #  match_gameweek tipini hafta olarak kayıt et
    eng_players_df["match_gameweek"] = eng_players_df["match_gameweek"].astype("int64")

    # Karışıklığı önemle için match_winner id ekle
    eng_players_df.rename(columns={"match_winner": "match_winner_id"}, inplace=True)

    # Eşisiz değer sayısı 2 nin altında olanları columsları sil
    unique_value_tow = [col for col in eng_players_df.columns if eng_players_df[col].nunique() < 2]
    eng_players_df.drop(unique_value_tow, axis=1, inplace=True)

    # Tarih değişkeni tip düzelme
    eng_players_df["match_dateutc"] = pd.to_datetime(eng_players_df["match_dateutc"])

    # match_winner_id hata düzeltmesi
    # 2 maçta beraberlik olmadığı halde match_winner_id=0 yazılmış,
    # skor karşılaştırmasıyla doğru winner'ı hesaplıyoruz
    def compute_winner(row):
        s1, s2 = row["match_team1.score"], row["match_team2.score"]
        if s1 == s2:
            return 0
        elif s1 > s2:
            return row["match_team1.teamId"]
        else:
            return row["match_team2.teamId"]

    eng_players_df["match_winner_id_fixed"] = eng_players_df.apply(compute_winner, axis=1)

    # unkow değerleri nan çevirdik
    eng_players_df["player_currentTeamId"] = eng_players_df["player_currentTeamId"].replace("unknow", np.nan)
    eng_players_df["player_foot"] = eng_players_df["player_foot"].replace("unknow", np.nan)

    if csv:
        eng_players_df.to_csv("eng_players_data_prep.csv",index=False)
    return eng_players_df


import pandas as pd

eng_df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Feature_Engineer\Data\event_england_merged.csv",low_memory=False)
df = event_eng_data_preprocessing(eng_df,True)
df = df[["eventName", "subEventName", "pos_orig_x", "pos_orig_y", "pos_dest_x", "pos_dest_y"]]

corner = df[df["subEventName"] == "Corner"]
at_corner_flag = corner["pos_orig_x"].isin([0, 100]) & corner["pos_orig_y"].isin([0, 100])
print("corner satiri:", len(corner))
print("koseden dogru baslayan oran:", at_corner_flag.mean())

throwin = df[df["subEventName"] == "Throw in"]
at_line = throwin["pos_orig_y"].isin([0, 100])
print("throw-in satiri:", len(throwin))
print("hat uzerinden dogru baslayan oran:", at_line.mean())





