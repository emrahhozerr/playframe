import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns",500)
pd.set_option("display.width",1000)
pd.set_option("expand_frame_repr",False)
pd.set_option("float_format",lambda x: "%.3f" % x)

eng_df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data Enginers\event_england_merged.csv",low_memory=False)
tags_df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data Enginers\event_tags_long.csv")

################################################
# eng_df EDA
################################################
eng_df.shape
eng_df.head()

# 1 olan değişkenler işin bittikten sonra silmeyi unutma eşisz değeri
# 1 tane eksik ayak bilgis var bunu oynucun diğer verilerine göre doldurlaım
# Eksiklikler
# bütün eksiklikler players_id = 0 dan dolayı kaynaklanır
# player_id 0 ise taç gibi bazı durumları belirtiyor bize
eng_df.isnull().sum().sort_values(ascending=False).head(20)

# type ayrımı yapıyoruz daha rahat eda yapmak için
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(eng_df)

eng_df[num_cols].head()
num_drop = ["eventId","playerId","matchId","teamId",
            "subEventId","id","match_team1.teamId","match_team2.teamId","match_winner"]
# match_winner kazanan takımın id sini veriyor
num_cols = [col for col in num_cols if col not in num_drop]

for col in cat_cols:
    print(col,eng_df[col].nunique())

drop_cat_cols= ["match_competitionId","team_area_team_conutry_id","match_competitionId"]
cat_cols = [col for col in cat_cols if col not in drop_cat_cols]

for col in cat_but_car:
    print(col,eng_df[col].nunique())


eng_df[num_cols].describe().T

#  match_gameweek tipini hafta olarak kayıt et #
# match_team1_own_goals ve match_team2_own_goals bir problem var kendi kalesine
# 18 gol atmak imkansız bu yüzden ana pipline düzeltme yap sil silme nedenimiz ana veride hatalı girilmiş bu kısımlar#

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

#for col in cat_cols:
    #cat_summary(eng_df,col,True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("##########################################")
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(eng_df, col, plot=True)

# 13 Mayıs 2018 en sonra tarih yaş hesaplarken unutma
# "eventSec" dk çevir #