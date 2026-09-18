import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns",500)
pd.set_option("display.width",1000)
pd.set_option("expand_frame_repr",False)
pd.set_option("float_format",lambda x: "%.3f" % x)

games_df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Feature_Engineer\Data\player_game_merged.csv")
tags_df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\EDA\Data\event_tags_long.csv")

games_df.head()
games_df.shape
games_df.dtypes
games_df.isnull().sum()

#  match_gameweek tipini hafta olarak kayıt et
games_df["match_gameweek"] = games_df["match_gameweek"].astype("int64")
games_df["match_gameweek"].dtypes

# Data columns tipini date yap
data_columsn = ["match_dateutc","birth_date"]

for col in data_columsn:
    games_df[col] = pd.to_datetime(games_df[col])

games_df[data_columsn].dtypes

# eşiz değeri 2 altında olanları siliyoruz
unique_value_toww = [col for col in games_df.columns if games_df[col].nunique() < 2]
games_df.drop(unique_value_toww,axis=1,inplace=True)

# Karışıklığı önemle için match_winner id ekle
games_df.rename(columns={"match_winner":"match_winner_id"},inplace=True)
games_df["match_winner_id"]

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

games_df["match_winner_id_fixed"] = games_df.apply(compute_winner, axis=1)
# kontrol: kaç satır düzeltildi
(games_df["match_winner_id"] != games_df["match_winner_id_fixed"]).sum()

# unkow değerleri nan çevirdik
unknow_col = [col for col in games_df.columns if (games_df[col].astype(str) == "unknow").any()]
games_df["player_info_currentTeamId"] = games_df["player_info_currentTeamId"].replace("unknow", np.nan)
games_df["player_info_foot"] = games_df["player_info_foot"].replace("unknow", np.nan)

def player_game_data_preprocessing(dataframe, csv=False):
    import pandas as pd
    import numpy as np

    data = dataframe.copy()

    #  match_gameweek tipini hafta olarak kayıt et
    data["match_gameweek"] = data["match_gameweek"].astype("int64")

    # Data columns tipini date yap
    data_columsn = ["match_dateutc", "birth_date"]
    for col in data_columsn:
        data[col] = pd.to_datetime(data[col])

    # eşsiz değeri 2 altında olanları siliyoruz
    unique_value_toww = [col for col in data.columns if data[col].nunique() < 2]
    data.drop(unique_value_toww, axis=1, inplace=True)

    # Karışıklığı önlemek için match_winner id ekle
    data.rename(columns={"match_winner": "match_winner_id"}, inplace=True)

    # match_winner_id hata düzeltmesi
    def compute_winner(row):
        s1, s2 = row["match_team1.score"], row["match_team2.score"]
        if s1 == s2:
            return 0
        elif s1 > s2:
            return row["match_team1.teamId"]
        else:
            return row["match_team2.teamId"]

    data["match_winner_id_fixed"] = data.apply(compute_winner, axis=1)

    # unknow değerlerini nan'a çevirdik
    data["player_info_currentTeamId"] = data["player_info_currentTeamId"].replace("unknow", np.nan)
    data["player_info_foot"] = data["player_info_foot"].replace("unknow", np.nan)

    if csv:
        data.to_csv("player_game_data_prep.csv", index=False)

    return data


import pandas as pd

games_df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Feature_Engineer\Data\player_game_merged.csv")

df = player_game_data_preprocessing(games_df,True)


