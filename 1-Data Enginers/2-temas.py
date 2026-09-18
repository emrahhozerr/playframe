import pandas as pd
import numpy as np
import ast

pd.set_option("display.width",1000)
pd.set_option("display.max_columns",500)
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.float_format",lambda x: "%.3f" % x)

df = pd.read_csv(r"Data/teamss.csv")

df.shape
df.head()
df.isnull().sum()
df.dtypes
df.nunique()
df.duplicated().sum()

# name yazıyor karışıklık olmasın diye team name yaptım oyuncu adında name yazıyor
df.rename(columns={"name":"team_name"},inplace=True)

# ingilizce formata gelidği için diğer dillerden geçerkken
# kelime yapaısnı bozmuş onu düzgün düzeltelim
df["team_name"] = df["team_name"].apply(lambda x: x.encode().decode("unicode_escape"))
df["officialName"] = df["officialName"].apply(lambda x: x.encode().decode("unicode_escape"))
df["city"] = df["city"].apply(lambda x: x.encode().decode("unicode_escape"))

# area değişkeni sözlük yapısı olarak tam onumamış onu düzeltelim
df["area"] = df["area"].apply(ast.literal_eval) #

# area sözlük yapısını değişkene veririyoruz
df["area_team_conutry"] = df["area"].apply(lambda x:x['name'])
df["area_team_conutry_id"] = df["area"].apply(lambda x:x['id'])

df.drop("area",axis=1,inplace=True)

# Columns Sırasını düzenleme
ordered_team_columns = [
    'wyId','team_name','officialName','type','city','area_team_conutry','area_team_conutry_id']

df_new= df[ordered_team_columns]

# Dışarıya aktarma kısmı
df_new.to_csv("teamss.csv",index=False)


def teams_pipline(dataframe):
    import pandas as pd
    import numpy as np
    import ast

    # name yazıyor karışıklık olmasın diye team name yaptım oyuncu adında name yazıyor
    dataframe.rename(columns={"name": "team_name"}, inplace=True)

    # ingilizce formata gelidği için diğer dillerden geçerkken
    # kelime yapaısnı bozmuş onu düzgün düzeltelim
    dataframe["team_name"] = dataframe["team_name"].apply(lambda x: x.encode().decode("unicode_escape"))
    dataframe["officialName"] = dataframe["officialName"].apply(lambda x: x.encode().decode("unicode_escape"))
    dataframe["city"] = dataframe["city"].apply(lambda x: x.encode().decode("unicode_escape"))

    # area değişkeni sözlük yapısı olarak tam onumamış onu düzeltelim
    dataframe["area"] = dataframe["area"].apply(ast.literal_eval)  #

    # area sözlük yapısını değişkene veririyoruz
    dataframe["area_team_conutry"] = dataframe["area"].apply(lambda x: x['name'])
    dataframe["area_team_conutry_id"] = dataframe["area"].apply(lambda x: x['id'])
    dataframe.drop("area", axis=1, inplace=True)

    # Columns Sırasını düzenleme
    ordered_team_columns = ['wyId', 'team_name', 'officialName', 'type', 'city', 'area_team_conutry', 'area_team_conutry_id']
    df_new = df[ordered_team_columns]
    df_new.to_csv("teamss.csv", index=False)
    return df_new

teams_pipline(df)


