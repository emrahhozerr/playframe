import pandas as pd
import numpy as np
import ast

pd.set_option("display.width",1000)
pd.set_option("display.max_columns",500)
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.float_format",lambda x: "%.3f" % x)


df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data\player_games.csv")
df.head()
df.shape
df.isnull().sum()
df.value_counts()
df["minutes_played"].describe([0.01,0.05,0.10,0.20,0.30,0.40]).T
df["jersey_number"].value_counts()
df.dtypes
df.duplicated().sum()

# Gereksiz Sütünlardan kurtul
del_cols = ['Unnamed: 0','firstname', 'lastname',"jersey_number"]
df.drop(del_cols,axis=1,inplace=True)

# shortName
df.rename(columns={"nickname":"shortName"},inplace=True)

# tarih değişkenin tipini düzelt
df["birth_date"] = pd.to_datetime(df["birth_date"])

# minutes_played değişkeninde 0 ve altı olan ifadeleri nan yap
df.loc[df["minutes_played"] <= 0,"minutes_played"] = np.nan

# 0 olan ları nan çevirmiştik şimdi onlaru kaldırıyorum
df.dropna(subset=["minutes_played"],axis=0,inplace=True)

# Sıralamsı
ordered_cols = [
    # 1. Anahtarlar ve Eşleşme ID'leri
    'game_id',"team_id","player_id",
    # 2. Oyuncu Tanımlayıcıları
    'player_name','shortName','birth_date',
    # 3. Maç İçi Metrikler / Durumlar
    'is_starter','minutes_played']

df_new = df[ordered_cols]

df_new.to_csv("players_game.csv",index=False)

def players_games_pipline(dataframe):
    import pandas as pd
    import numpy as np
    import ast

    # Gereksiz Sütünlardan kurtul
    del_cols = ['Unnamed: 0', 'firstname', 'lastname', "jersey_number"]
    dataframe.drop(del_cols, axis=1, inplace=True)

    # shortName
    dataframe.rename(columns={"nickname": "shortName"}, inplace=True)

    # tarih değişkenin tipini düzelt
    dataframe["birth_date"] = pd.to_datetime(dataframe["birth_date"])

    # minutes_played değişkeninde 0 ve altı olan ifadeleri nan yap
    dataframe.loc[dataframe["minutes_played"] <= 0, "minutes_played"] = np.nan

    # 0 olan ları nan çevirmiştik şimdi onlaru kaldırıyorum
    dataframe.dropna(subset=["minutes_played"], axis=0, inplace=True)

    # Sıralamsı
    ordered_cols = [
        # 1. Anahtarlar ve Eşleşme ID'leri
        'game_id', "team_id", "player_id",
        # 2. Oyuncu Tanımlayıcıları
        'player_name', 'shortName', 'birth_date',
        # 3. Maç İçi Metrikler / Durumlar
        'is_starter', 'minutes_played']

    df_new = dataframe[ordered_cols]
    df_new.to_csv("players_gamee.csv", index=False)
    return df_new

players_games_pipline(df)



df[df['player_id'] == 0]




