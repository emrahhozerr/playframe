import pandas as pd
import numpy as np
import warnings

pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

team_match = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\htactic_builder\team_match_tactics.csv")


# oyuncu tarafiyla ayni mantik: yuzde kolonlarini dogrudan ortalamak yerine
# ham sayaclari sezon uzerinden topluyoruz, yuzdeleri o toplamlardan yeniden
# hesapliyoruz. esik/eligibility filtresine gerek yok - butun takimlar ayni
# sezonda ayni sayida (38) mac oynuyor
raw_count_cols = ["passes_total", "passes_accurate", "long_balls", "final_third_entries",
                  "box_entries", "wide_actions", "defensive_duels", "high_press_actions",
                  "set_pieces", "shots_total", "goals",]

season_sum = team_match.groupby("teamId", as_index=False)[raw_count_cols].sum()

matches_played = (team_match.groupby("teamId", as_index=False)["matchId"].nunique()
                  .rename(columns={"matchId": "matches_played"}))

# ortalama mesafe/ilerleme gibi kolonlar toplanamaz, mac uzerinden ortalama aliniyor
avg_cols = ["avg_pass_distance_m", "avg_forward_progress"]
season_avg = team_match.groupby("teamId", as_index=False)[avg_cols].mean()

season = season_sum.merge(matches_played, on="teamId").merge(season_avg, on="teamId")

# yuzde metriklerini sezon toplamlarindan yeniden hesapla
season["pass_accuracy_pct"] = np.where(season["passes_total"] > 0,
    season["passes_accurate"] / season["passes_total"] * 100, np.nan)
season["long_ball_pct"] = np.where(season["passes_total"] > 0,
    season["long_balls"] / season["passes_total"] * 100, np.nan)
season["wide_action_pct"] = np.where(season["passes_total"] > 0,
    season["wide_actions"] / season["passes_total"] * 100, np.nan)
season["high_press_pct"] = np.where(season["defensive_duels"] > 0,
    season["high_press_actions"] / season["defensive_duels"] * 100, np.nan)

# mac basina oranlar
for col in raw_count_cols:
    season[f"{col}_per_match"] = season[col] / season["matches_played"]

# z-score / percentile - oyuncudan farkli olarak pozisyon gibi bir alt grup yok,
# butun takimlar tek grup olarak kiyaslaniyor
per_match_cols = [col for col in season.columns if col.endswith("_per_match")]
pct_metric_cols = [col for col in season.columns if col.endswith("_pct")]
z_source_cols = per_match_cols + pct_metric_cols + avg_cols

norm = season.copy()
for col in z_source_cols:
    norm[f"{col}_z"] = (norm[col] - norm[col].mean()) / norm[col].std()
    norm[f"{col}_percentile"] = norm[col].rank(pct=True) * 100

season.to_csv("team_season_tactics.csv", index=False)
norm.to_csv("team_season_tactics_normalized.csv", index=False)

# Fonksiyon
def tactic_season_normalize_func(team_match, csv=False):
    import pandas as pd
    import numpy as np

    team_match = team_match

    # oyuncu tarafiyla ayni mantik: yuzde kolonlarini dogrudan ortalamak yerine
    # ham sayaclari sezon uzerinden topluyoruz, yuzdeleri o toplamlardan yeniden
    # hesapliyoruz. esik/eligibility filtresine gerek yok - butun takimlar ayni
    # sezonda ayni sayida (38) mac oynuyor
    raw_count_cols = [
        "passes_total", "passes_accurate", "long_balls", "final_third_entries",
        "box_entries", "wide_actions", "defensive_duels", "high_press_actions",
        "set_pieces", "shots_total", "goals",
    ]

    season_sum = team_match.groupby("teamId", as_index=False)[raw_count_cols].sum()

    matches_played = (
        team_match.groupby("teamId", as_index=False)["matchId"].nunique()
        .rename(columns={"matchId": "matches_played"})
    )

    # ortalama mesafe/ilerleme gibi kolonlar toplanamaz, mac uzerinden ortalama aliniyor
    avg_cols = ["avg_pass_distance_m", "avg_forward_progress"]
    season_avg = team_match.groupby("teamId", as_index=False)[avg_cols].mean()

    season = season_sum.merge(matches_played, on="teamId").merge(season_avg, on="teamId")

    # yuzde metriklerini sezon toplamlarindan yeniden hesapla
    season["pass_accuracy_pct"] = np.where(season["passes_total"] > 0,
        season["passes_accurate"] / season["passes_total"] * 100, np.nan)
    season["long_ball_pct"] = np.where(season["passes_total"] > 0,
        season["long_balls"] / season["passes_total"] * 100, np.nan)
    season["wide_action_pct"] = np.where(season["passes_total"] > 0,
        season["wide_actions"] / season["passes_total"] * 100, np.nan)
    season["high_press_pct"] = np.where(season["defensive_duels"] > 0,
        season["high_press_actions"] / season["defensive_duels"] * 100, np.nan)

    # mac basina oranlar
    for col in raw_count_cols:
        season[f"{col}_per_match"] = season[col] / season["matches_played"]

    # z-score / percentile - oyuncudan farkli olarak pozisyon gibi bir alt grup yok,
    # butun takimlar tek grup olarak kiyaslaniyor
    per_match_cols = [col for col in season.columns if col.endswith("_per_match")]
    pct_metric_cols = [col for col in season.columns if col.endswith("_pct")]
    z_source_cols = per_match_cols + pct_metric_cols + avg_cols

    norm = season.copy()
    for col in z_source_cols:
        norm[f"{col}_z"] = (norm[col] - norm[col].mean()) / norm[col].std()
        norm[f"{col}_percentile"] = norm[col].rank(pct=True) * 100

    if csv:
        season.to_csv("team_season_tactics.csv", index=False)
        norm.to_csv("team_season_tactics_normalized.csv", index=False)

    return season, norm