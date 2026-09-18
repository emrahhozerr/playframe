import pandas as pd
import numpy as np
import ast


pd.set_option("display.width",1000)
pd.set_option("display.max_columns",500)
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.float_format",lambda x: "%.3f" % x)

df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data\players.csv")
df.head()
df.info()
df.shape
df.isnull().sum().sort_values(ascending=False)
df.describe().T
df.nunique()
df.duplicated().sum()


##########################################
# Sözlük Halindeki değişkenleri düzelt #
###########################################
# String halindeki dict'i gerçek dict'e çevir
df["role"] = df["role"].apply(ast.literal_eval)
df["passportArea"] = df["passportArea"].apply(ast.literal_eval)
df["birthArea"] = df["birthArea"].apply(ast.literal_eval)

# role sütunundan sadece pozisyon kodunu çıkar (GK, DF, MD, FW gibi)
df["role_code"] = df["role"].apply(lambda x: x["code2"])
df["role_name"] = df["role"].apply(lambda x: x["name"])

# passportArea'dan ülke adını çıkar
df["passport_country"] = df["passportArea"].apply(lambda x: x["name"])
df["passport_country_id"] = df["passportArea"].apply(lambda x: x['id'])

# birthArea'dan doğum yeri ülkesini çıkar
df["birth_country"] = df["birthArea"].apply(lambda x: x["name"])
df["birth_country_id"] = df["birthArea"].apply(lambda x: x["id"])

# Sözlük yapılarından değişken üretiğimiz için  artık bu değişken
# ile işimiz olmadığı için sildim
df.drop(["role", "passportArea", "birthArea"], axis=1, inplace=True)

##########################################
# Aykırı Değerler #
###########################################
# 0 olması gerek değerlere i nan yap
df[["weight","height"]].describe().T
df["weight"] = df["weight"].replace(0,np.nan)
df["height"] = df["height"].replace(0,np.nan)

##########################################
# Nan Değerler #
###########################################
# Milli takımıda boşluklaru "unknow" yaptım oyuncu mili takıma seçilmemiş
df["currentNationalTeamId"] = df["currentNationalTeamId"].fillna("unknow")

# takım id sinide unkow yapcam belki oyuncu o yıl takımsız kaldı yada genç oyuncu
df["currentTeamId"] = df["currentTeamId"].fillna("unknow")

# BOY VE kiloda o mevkiye göre median alıp oan göre doldurma yapcam median
# olmasının sebebi sporcu verilerin normal bir davranış sergilememesinden dolayı

# a) boy nan değerler
df.groupby('role_code')['height'].median()
df['height'] = df.groupby('role_code')['height'].transform(
    lambda x: x.fillna(x.median()))
# b) kilo nan değeler
df.groupby('role_code')['weight'].median()
df['weight'] = df.groupby('role_code')['weight'].transform(
    lambda x: x.fillna(x.median()))

# hangi ayağı kullandğı bilgi eksik kişiler var "unknow" olarak
# değiştirildi
df["foot"] = df["foot"].fillna("unknow")

##########################################
# Tarih Verilerinin Tipini Düzeltme #
###########################################
df.dtypes
df["birthDate"] = pd.to_datetime(df["birthDate"])

#######################################################
# Gereksiz Değişkenlerden Kurtulma,Oluşturma,Düzeltme #
#######################################################
# İsim ve Soy isimleri birleştir
df["name"] = df["firstName"] + " " + df["lastName"]
df.drop(["firstName","lastName"],axis=1,inplace=True)

# isimlerdeki bozukluğu gider
df["name"] = df["name"].apply(lambda x: x.encode().decode("unicode_escape"))
df["shortName"] = df["shortName"].apply(lambda x: x.encode().decode("unicode_escape"))

# oyuncu adını "player_name olacak sonra yap"
df.rename(columns={"name":"player_name",},inplace=True)

#####################
# Data Frame Sırlama #
#####################
# Profesyonel ve mantıksal sütun sıralaması
ordered_columns = [
    # 1. Kimlik
    "wyId","player_name","shortName",
    # 2. Rol ve Takım
    "role_code","role_name","currentTeamId",
    # 3. Fiziksel ve Biyolojik
    "birthDate","height","weight","foot",
    # 4. Ülke Bilgileri
     "birth_country","passport_country",
]

df_new = df[ordered_columns]

# dışarı aktrama temiz dosyayı
df_new.to_csv("players.csv",index=False)

"""
BİLGİ ="foot","currentTeamId","currentNationalTeamId" bu üç değişkenin
eksik değişrleri "unknow" ile doldurdu 
"""

# Fonksiyonlarştırma
def players_pipline(dataframe):
    """
    BİLGİ ="foot","currentTeamId","currentNationalTeamId" bu üç değişkenin
    eksik değişrleri "unknow" ile doldurdu
    """
    import pandas as pd
    import numpy as np
    import ast

    # Sözlük Halindeki değişkenleri düzelt #

    # String halindeki dict'i gerçek dict'e çevir
    dataframe["role"] = dataframe["role"].apply(ast.literal_eval)
    dataframe["passportArea"] = dataframe["passportArea"].apply(ast.literal_eval)
    dataframe["birthArea"] = dataframe["birthArea"].apply(ast.literal_eval)

    # role sütunundan sadece pozisyon kodunu çıkar (GK, dataframe, MD, FW gibi)
    dataframe["role_code"] = dataframe["role"].apply(lambda x: x["code2"])
    dataframe["role_name"] = dataframe["role"].apply(lambda x: x["name"])

    # passportArea'dan ülke adını çıkar
    dataframe["passport_country"] = dataframe["passportArea"].apply(lambda x: x["name"])
    dataframe["passport_country_id"] = dataframe["passportArea"].apply(lambda x: x['id'])

    # birthArea'dan doğum yeri ülkesini çıkar
    dataframe["birth_country"] = dataframe["birthArea"].apply(lambda x: x["name"])
    dataframe["birth_country_id"] = dataframe["birthArea"].apply(lambda x: x["id"])

    # Sözlük yapılarından değişken üretiğimiz için  artık bu değişken
    # ile işimiz olmadığı için sildim
    dataframe.drop(["role", "passportArea", "birthArea"], axis=1, inplace=True)
    ###########################################################################
    # Aykırı Değerler

    # 0 olması gerek değerlere i nan yap
    dataframe["weight"] = dataframe["weight"].replace(0, np.nan)
    dataframe["height"] = dataframe["height"].replace(0, np.nan)
    ##########################################################################
    # Nan Değerler #

    # Milli takımıda boşluklaru "unknow" yaptım oyuncu mili takıma seçilmemiş
    dataframe["currentNationalTeamId"] = dataframe["currentNationalTeamId"].fillna("unknow")

    # takım id sinide unkow yapcam belki oyuncu o yıl takımsız kaldı yada genç oyuncu
    dataframe["currentTeamId"] = dataframe["currentTeamId"].fillna("unknow")

    # BOY VE kiloda o mevkiye göre median alıp oan göre doldurma yapcam median
    # olmasının sebebi sporcu verilerin normal bir davranış sergilememesinden dolayı
    # a) boy nan değerler
    dataframe['height'] = dataframe.groupby('role_code')['height'].transform(
        lambda x: x.fillna(x.median()))

    # b) kilo nan değeler
    dataframe['weight'] = dataframe.groupby('role_code')['weight'].transform(
        lambda x: x.fillna(x.median()))

    # hangi ayağı kullandğı bilgi eksik kişiler var "unknow" olarak
    # değiştirildi
    dataframe["foot"] = dataframe["foot"].fillna("unknow")
    ##########################################################################
    # Tarih Verilerinin Tipini Düzeltme #
    dataframe["birthDate"] = pd.to_datetime(dataframe["birthDate"])
    ##########################################################################
    # Gereksiz Değişkenlerden Kurtulma,Oluşturma,Düzeltme #

    # İsim ve Soy isimleri birleştir
    dataframe["name"] = dataframe["firstName"] + " " + dataframe["lastName"]
    dataframe.drop(["firstName", "lastName"], axis=1, inplace=True)

    # isimlerdeki bozukluğu gider
    dataframe["name"] = dataframe["name"].apply(lambda x: x.encode().decode("unicode_escape"))
    dataframe["shortName"] = dataframe["shortName"].apply(lambda x: x.encode().decode("unicode_escape"))

    # oyuncu adını "player_name olacak sonra yap"
    dataframe.rename(columns={"name": "player_name", }, inplace=True)
    #############################################################################
    # Data Frame Sırlama #
    # Profesyonel ve mantıksal sütun sıralaması
    ordered_columns = [
        # 1. Kimlik
        "wyId", "player_name", "shortName",
        # 2. Rol ve Takım
        "role_code", "role_name", "currentTeamId",
        # 3. Fiziksel ve Biyolojik
        "birthDate", "height", "weight", "foot",
        # 4. Ülke Bilgileri
        "birth_country", "passport_country",
    ]

    df_new = dataframe[ordered_columns]
    #df_new.to_csv("playerss.csv", index=False)
    return df_new

df_new = players_pipline(df)

df_new[df_new['wyId'] == 0]











