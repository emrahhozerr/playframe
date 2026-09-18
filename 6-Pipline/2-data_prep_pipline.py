
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

EVENT_MERGED_CSV = r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\EDA\Data\event_england_merged.csv"
PLAYER_GAME_MERGED_CSV = r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\EDA\Data\player_game_merged.csv"


def main():
    print("1/2 - event_eng_data_preprocessing çalışıyor...")
    event_df = pd.read_csv(EVENT_MERGED_CSV,low_memory=False)
    event_eng_data_preprocessing(event_df, csv=True)
    print("   -> eng_players_data_prep.csv oluşturuldu")

    print("2/2 - player_game_data_preprocessing çalışıyor...")
    player_game_df = pd.read_csv(PLAYER_GAME_MERGED_CSV)
    player_game_data_preprocessing(player_game_df, csv=True)
    print("   -> player_game_data_prep.csv oluşturuldu")

    print("\nÖn işleme tamamlandı.")


if __name__ == "__main__":
    main()