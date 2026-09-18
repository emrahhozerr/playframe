import pandas as pd
df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\5-Pipline\data\player_season_normalized.csv")
print(df["league_code"].value_counts())