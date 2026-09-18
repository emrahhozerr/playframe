import pandas as pd
import numpy as np
import warnings

pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("expand_frame_repr", False)
pd.set_option("float_format", lambda x: "%.3f" % x)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

per90 = pd.read_csv(
    r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Feature_Engineer\Player_Features\Data\perc90_builder.csv")
per90.shape


# xt_added'i per90 tablosuna playerId+matchId uzerinden bagla
# xt'si olmayan move aksiyonu hic yapmamis oyuncu-mac satirlari icin 0 kabul ediyoruz
xt_match = pd.read_csv(
    r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Feature_Engineer\Player_Features\Data\event_xt.csv")

per90 = per90.merge(xt_match, on=["playerId", "matchId"], how="left")
per90["xt_added_total"] = per90["xt_added_total"].fillna(0)
print("xt merge sonrasi shape:", per90.shape)

# oyuncu-sezon agregasyonu

df = per90.copy()

######################################3
# oyuncu-mac bazli per90 tablosunu oyuncu-sezon bazina indirgiyoruz.
# per90/yuzde kolonlarini dogrudan ortalamak yanlis olur çünkü kucuk dakikali
# maclarin orani carpitir - onun yerine ham sayaclari ve dakikayi sezon
# uzerinden toplayip, per90/yuzde metriklerini o toplamlardan yeniden
# hesapliyoruz.

meta_cols = ["playerId", "matchId", "teamId", "player_name", "is_starter",
             "minutes_played", "player_info_role_code", "player_info_role_name",
             "match_gameweek", "player_info_height", "player_info_weight",
             "player_info_foot", "player_info_birth_country", "player_info_passport_country",
             "player_info_currentTeamId", "team_team_name", "team_city", "team_area_team_conutry",
             "match_dateutc", "match_venue", "player_age_at_match", "opponent_team_id","is_home",
             "team_goals", "opponent_goals", "team_goals_HT", "opponent_goals_HT","match_result",]

p90_cols = [col for col in df.columns if col.endswith("_p90")]
pct_cols = [col for col in df.columns if col.endswith("_pct")]
raw_count_cols = [col for col in df.columns if col not in meta_cols + p90_cols + pct_cols]
len(raw_count_cols)

# oyuncu basina sezon boyunca sabit kalan meta bilgiler
player_meta = df.groupby("playerId", as_index=False).agg(
    player_name=("player_name", "first"),
    role_code=("player_info_role_code", "first"),
    role_name=("player_info_role_name", "first"),
    height=("player_info_height", "first"),
    weight=("player_info_weight", "first"),
    foot=("player_info_foot", "first"),
    birth_country=("player_info_birth_country", "first"),
    current_team_id=("player_info_currentTeamId", "last"))

player_meta.shape

# sezon toplamı yani ham sayaclar + dakika
season_sum = df.groupby("playerId", as_index=False)[raw_count_cols + ["minutes_played"]].sum()

# oyuncunun sezonda kac farkli mac oynadigi yani satir sayisi degil benzersiz mac sayısı
matches_played = (
    df.groupby("playerId", as_index=False)["matchId"].nunique().rename(columns={
        "matchId": "matches_played"}))

season = player_meta.merge(season_sum, on="playerId").merge(matches_played, on="playerId")

season.shape

# sezon per90'i yeniden hesaplama mac bazli per90'lari ortalamak yerine kullancaz
minutes = season["minutes_played"].replace(0, np.nan)
for col in raw_count_cols:
    season[f"{col}_p90"] = season[col] / minutes * 90

# yuzde metriklerini sezon toplamlarindan yeniden hesapla
season["pass_accuracy_pct"] = np.where(season["passes_total"] > 0,
                                       season["passes_accurate"] / season["passes_total"] * 100, np.nan)

season["overall_accuracy_pct"] = np.where((season["accurate_actions_total"] + season["inaccurate_actions_total"]) > 0,
    season["accurate_actions_total"] / (season["accurate_actions_total"] + season["inaccurate_actions_total"]) * 100,
    np.nan)

season["duel_win_pct"] = np.where(season["duels_total"] > 0, season["duels_won"] / season["duels_total"] * 100, np.nan)
season["duel_loss_pct"] = np.where(season["duels_total"] > 0, season["duels_lost"] / season["duels_total"] * 100, np.nan)
season["duel_neutral_pct"] = np.where(season["duels_total"] > 0, season["duels_neutral"] / season["duels_total"] * 100, np.nan)
season["take_on_success_pct"] = np.where(season["take_ons"] > 0, season["take_ons_won"] / season["take_ons"] * 100, np.nan)
season["shot_accuracy_pct"] = np.where(season["shots_total"] > 0, season["shots_on_target"] / season["shots_total"] * 100, np.nan)
season["shot_conversion_pct"] = np.where(season["shots_total"] > 0, season["goals"] / season["shots_total"] * 100, np.nan)
foot_total = season["left_foot_actions"] + season["right_foot_actions"]
season["right_foot_dominance_pct"] = np.where(foot_total > 0, season["right_foot_actions"] / foot_total * 100, np.nan)

season.shape

#  oyuncu sayisi kaybolmus mu yani per90'daki benzersiz oyuncu sayisiyla ayni oldumu
per90["playerId"].nunique()
season["playerId"].nunique()

# grup istatistikleri ortalama/std/persentil referansi sadece 600 dakika
# eşiğini gecen oyunculardan hesaplanir az dakikali oyuncularin gurultusu yani
# orn. 10 dakikada 1 gol = per90'da cok yuksek deger grubun ortalamasini
# bozmasin diye. ama esigin altindaki oyunculara da referans grubuna gore
# bir skor atanir, tablodan cikarilmazlar.

group_cols = ["role_code"]

norm = season.copy()
norm_p90_cols = [col for col in norm.columns if col.endswith("_p90")]
norm["is_eligible"] = norm["minutes_played"] >= 600
norm["is_eligible"].sum(), "/", len(norm)

eligible = norm[norm["is_eligible"]]

# z-score: grup ortalama/std'sini sadece esik ustu oyunculardan hesapla
# sonra herkese esik alti dahil uygula
group_stats = eligible.groupby(group_cols)[norm_p90_cols].agg(["mean", "std"])
group_stats.columns = [f"{col}_{stat}" for col, stat in group_stats.columns]
norm = norm.merge(group_stats, on=group_cols, how="left")

for col in norm_p90_cols:
    norm[f"{col}_z"] = (norm[col] - norm[f"{col}_mean"]) / norm[f"{col}_std"]

# yardimci mean/std kolonlarini temizle sadece z-score kalsin
norm = norm.drop(columns=[col for col in norm.columns if col.endswith("_mean") or col.endswith("_std")])

# percentile: sadece esik ustu oyuncular arasinda mevki bazli siralama
mask = norm["is_eligible"]
for col in norm_p90_cols:
    norm[f"{col}_percentile"] = np.nan
    norm.loc[mask, f"{col}_percentile"] = norm.loc[mask].groupby(group_cols)[col].rank(pct=True) * 100

norm.shape

# forvetler arasinda gol/90 persentiline gore ilk 5 gözlemi getir doğru olmuş mu bakalım
(norm[norm["role_code"] == "FW"].sort_values("goals_p90_percentile", ascending=False)
 [["player_name", "minutes_played", "goals_p90", "goals_p90_percentile"]].head())

# kaydet
season.to_csv("player_season.csv", index=False)
norm.to_csv("player_season_normalized.csv", index=False)










def season_normalize_func(per90_1,xt_match,csv=False):
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    per90 = per90_1

    # xt_added'i per90 tablosuna playerId+matchId uzerinden bagla
    # xt'si olmayan move aksiyonu hic yapmamis oyuncu-mac satirlari icin 0 kabul ediyoruz
    xt_match = xt_match
    per90 = per90.merge(xt_match, on=["playerId", "matchId"], how="left")
    per90["xt_added_total"] = per90["xt_added_total"].fillna(0)

    # oyuncu-mac bazli per90 tablosunu oyuncu-sezon bazina indirgiyoruz.
    # per90/yuzde kolonlarini dogrudan ortalamak yanlis olur çünkü kucuk dakikali
    # maclarin orani carpitir - onun yerine ham sayaclari ve dakikayi sezon
    # uzerinden toplayip, per90/yuzde metriklerini o toplamlardan yeniden
    # hesapliyoruz.

    df = per90.copy()

    meta_cols = ["playerId", "matchId", "teamId", "player_name", "is_starter",
                 "minutes_played", "player_info_role_code", "player_info_role_name",
                 "match_gameweek", "player_info_height", "player_info_weight",
                 "player_info_foot", "player_info_birth_country", "player_info_passport_country",
                 "player_info_currentTeamId", "team_team_name", "team_city", "team_area_team_conutry",
                 "match_dateutc", "match_venue", "player_age_at_match", "opponent_team_id", "is_home",
                 "team_goals", "opponent_goals", "team_goals_HT", "opponent_goals_HT", "match_result", ]

    p90_cols = [col for col in df.columns if col.endswith("_p90")]
    pct_cols = [col for col in df.columns if col.endswith("_pct")]
    raw_count_cols = [col for col in df.columns if col not in meta_cols + p90_cols + pct_cols]

    # oyuncu basina sezon boyunca sabit kalan meta bilgiler
    player_meta = df.groupby("playerId", as_index=False).agg(
        player_name=("player_name", "first"),
        role_code=("player_info_role_code", "first"),
        role_name=("player_info_role_name", "first"),
        height=("player_info_height", "first"),
        weight=("player_info_weight", "first"),
        foot=("player_info_foot", "first"),
        birth_country=("player_info_birth_country", "first"),
        current_team_id=("player_info_currentTeamId", "last"))

    # sezon toplamı yani ham sayaclar + dakika
    season_sum = df.groupby("playerId", as_index=False)[raw_count_cols + ["minutes_played"]].sum()

    # oyuncunun sezonda kac farkli mac oynadigi yani satir sayisi degil benzersiz mac sayısı
    matches_played = (
        df.groupby("playerId", as_index=False)["matchId"].nunique().rename(columns={
            "matchId": "matches_played"}))

    season = player_meta.merge(season_sum, on="playerId").merge(matches_played, on="playerId")

    # sezon per90'i yeniden hesaplama mac bazli per90'lari ortalamak yerine kullancaz
    minutes = season["minutes_played"].replace(0, np.nan)
    for col in raw_count_cols:
        season[f"{col}_p90"] = season[col] / minutes * 90

    # yuzde metriklerini sezon toplamlarindan yeniden hesapla
    season["pass_accuracy_pct"] = np.where(season["passes_total"] > 0,
                                           season["passes_accurate"] / season["passes_total"] * 100, np.nan)

    season["overall_accuracy_pct"] = np.where(
        (season["accurate_actions_total"] + season["inaccurate_actions_total"]) > 0,
        season["accurate_actions_total"] / (
                    season["accurate_actions_total"] + season["inaccurate_actions_total"]) * 100,
        np.nan)

    season["duel_win_pct"] = np.where(season["duels_total"] > 0, season["duels_won"] / season["duels_total"] * 100,
                                      np.nan)
    season["duel_loss_pct"] = np.where(season["duels_total"] > 0, season["duels_lost"] / season["duels_total"] * 100,
                                       np.nan)
    season["duel_neutral_pct"] = np.where(season["duels_total"] > 0,
                                          season["duels_neutral"] / season["duels_total"] * 100, np.nan)
    season["take_on_success_pct"] = np.where(season["take_ons"] > 0, season["take_ons_won"] / season["take_ons"] * 100,
                                             np.nan)
    season["shot_accuracy_pct"] = np.where(season["shots_total"] > 0,
                                           season["shots_on_target"] / season["shots_total"] * 100, np.nan)
    season["shot_conversion_pct"] = np.where(season["shots_total"] > 0, season["goals"] / season["shots_total"] * 100,
                                             np.nan)
    foot_total = season["left_foot_actions"] + season["right_foot_actions"]
    season["right_foot_dominance_pct"] = np.where(foot_total > 0, season["right_foot_actions"] / foot_total * 100,np.nan)

    # grup istatistikleri ortalama/std/persentil referansi sadece 600 dakika
    # eşiğini gecen oyunculardan hesaplanir az dakikali oyuncularin gurultusu yani
    # orn. 10 dakikada 1 gol = per90'da cok yuksek deger grubun ortalamasini
    # bozmasin diye. ama esigin altindaki oyunculara da referans grubuna gore
    # bir skor atanir, tablodan cikarilmazlar.

    group_cols = ["role_code"]
    norm = season.copy()
    norm_p90_cols = [col for col in norm.columns if col.endswith("_p90")]
    norm["is_eligible"] = norm["minutes_played"] >= 600
    eligible = norm[norm["is_eligible"]]

    # z-score: grup ortalama/std'sini sadece esik ustu oyunculardan hesapla
    # sonra herkese esik alti dahil uygula
    group_stats = eligible.groupby(group_cols)[norm_p90_cols].agg(["mean", "std"])
    group_stats.columns = [f"{col}_{stat}" for col, stat in group_stats.columns]
    norm = norm.merge(group_stats, on=group_cols, how="left")

    for col in norm_p90_cols:
        norm[f"{col}_z"] = (norm[col] - norm[f"{col}_mean"]) / norm[f"{col}_std"]

    # yardimci mean/std kolonlarini temizle sadece z-score kalsin
    norm = norm.drop(columns=[col for col in norm.columns if col.endswith("_mean") or col.endswith("_std")])

    # percentile: sadece esik ustu oyuncular arasinda mevki bazli siralama
    mask = norm["is_eligible"]
    for col in norm_p90_cols:
        norm[f"{col}_percentile"] = np.nan
        norm.loc[mask, f"{col}_percentile"] = norm.loc[mask].groupby(group_cols)[col].rank(pct=True) * 100


    if csv:
        season.to_csv("player_season.csv", index=False)
        norm.to_csv("player_season_normalized.csv", index=False)

    return season,norm
