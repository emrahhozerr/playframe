import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

event_eng = r"C:\Users\ahmet\Masaüstü\ds\Miull Data Sciences Final Project\Feature_Engineer\Data\event_england_merged.csv"

usecols = ["id", "playerId", "matchId", "teamId", "eventName", "subEventName","pos_orig_x",
           "pos_orig_y", "pos_dest_x", "pos_dest_y","tag_101", "tag_703", "tag_1801"]

ev = pd.read_csv(event_eng, usecols=usecols)
ev.shape


# grid saha 16 uzunluk x ile 12 genislik, y ile bolgeye ayriliyor

n_x = 16
n_y = 12
n_zones = n_x * n_y

ev["orig_zone_x"] = np.minimum((ev["pos_orig_x"] // (100 / n_x)).astype(int), n_x - 1)
ev["orig_zone_y"] = np.minimum((ev["pos_orig_y"] // (100 / n_y)).astype(int), n_y - 1)
ev["dest_zone_x"] = np.minimum((ev["pos_dest_x"].fillna(-1) // (100 / n_x)).astype(int), n_x - 1)
ev["dest_zone_y"] = np.minimum((ev["pos_dest_y"].fillna(-1) // (100 / n_y)).astype(int), n_y - 1)

ev["orig_zone"] = ev["orig_zone_y"] * n_x + ev["orig_zone_x"]
ev["dest_zone"] = ev["dest_zone_y"] * n_x + ev["dest_zone_x"]


# 2) ilerletme yani move ve sut aksiyonlarini belirle

# ilerletme = isabetli pas + basarili dripling ground attacking duel, won gibi
is_pass_ok = (ev["eventName"] == "Pass") & (ev["tag_1801"] == 1)
is_take_on_ok = (ev["subEventName"] == "Ground attacking duel") & (ev["tag_703"] == 1)
is_move = is_pass_ok | is_take_on_ok

is_shot = ev["eventName"] == "Shot"
is_goal = is_shot & (ev["tag_101"] == 1)

is_move.sum()
is_shot.sum()
is_goal.sum()


# her bolge icin sut gol hareket sayaclarini cikar

move_counts = ev.loc[is_move, "orig_zone"].value_counts().reindex(range(n_zones), fill_value=0)
shot_counts = ev.loc[is_shot, "orig_zone"].value_counts().reindex(range(n_zones), fill_value=0)
goal_counts = ev.loc[is_goal, "orig_zone"].value_counts().reindex(range(n_zones), fill_value=0)

total_actions = move_counts + shot_counts

# o bolgeden sut cekilme olasiligi ile ilerletme yapilma olasiligi ve sutun gol olma olasiligi
shot_prob = np.where(total_actions > 0, shot_counts / total_actions, 0)
move_prob = 1 - shot_prob
goal_prob = np.where(shot_counts > 0, goal_counts / shot_counts, 0)


# gecis matrisi ile her bolgeden yapilan ilerletmelerin nereye gittigi

transition_counts = (ev.loc[is_move].groupby(["orig_zone", "dest_zone"]).size()
    .unstack(fill_value=0).reindex(index=range(n_zones), columns=range(n_zones), fill_value=0))

row_sums = transition_counts.sum(axis=1).replace(0, np.nan)
transition_matrix = transition_counts.div(row_sums, axis=0).fillna(0).values


# xT degerlerini iteratif hesapla
# her bolgenin degeri o bolgeden gol atma ihtimali + ilerletme yapip gittigi
# bolgelerin degerlerinin agirlikli ortalamasi ve baslangicta hepsi 0, birkac
# iterasyonda durağanlasiyor yani yakinsiyor.

xt = np.zeros(n_zones)
n_iterations = 10
for i in range(n_iterations):
    xt = shot_prob * goal_prob + move_prob * (transition_matrix @ xt)
    print(f"iterasyon {i + 1}, ortalama xt: {xt.mean():.5f}, max xt: {xt.max():.5f}")

xt_grid = xt.reshape(n_y, n_x)
print("xt grid (satirlar=y, kolonlar=x, hucum yonu sag taraf):")
print(pd.DataFrame(xt_grid).round(4))


# her ilerletme aksiyonu icin xt_added hesapla
# xt_added = bitis bolgesinin degeri - baslangic bolgesinin degeri.
# sadece "move" aksiyonlari icin anlamli, digerlerinde NaN kalir.

ev["xt_added"] = np.nan
ev.loc[is_move, "xt_added"] = xt[ev.loc[is_move, "dest_zone"].values] - xt[ev.loc[is_move, "orig_zone"].values]

ev.loc[is_move, "xt_added"].mean()


# oyuncu-mac bazinda topla, kaydet

xt_match = (ev.loc[is_move].groupby(["playerId", "matchId"], as_index=False)["xt_added"].sum()
    .rename(columns={"xt_added": "xt_added_total"}))
xt_match.shape
xt_match.sort_values("xt_added_total", ascending=False).head(10)

xt_match.to_csv("event_xt.csv", index=False)


def xt_bulider_func(event_eng,csv=False):
    import pandas as pd
    import numpy as np

    usecols = ["id", "playerId", "matchId", "teamId", "eventName", "subEventName", "pos_orig_x",
               "pos_orig_y", "pos_dest_x", "pos_dest_y", "tag_101", "tag_703", "tag_1801"]

    ev = pd.read_csv(event_eng, usecols=usecols)

    # grid saha 16 uzunluk x ile 12 genislik, y ile bolgeye ayriliyor
    n_x = 16
    n_y = 12
    n_zones = n_x * n_y

    ev["orig_zone_x"] = np.minimum((ev["pos_orig_x"] // (100 / n_x)).astype(int), n_x - 1)
    ev["orig_zone_y"] = np.minimum((ev["pos_orig_y"] // (100 / n_y)).astype(int), n_y - 1)
    ev["dest_zone_x"] = np.minimum((ev["pos_dest_x"].fillna(-1) // (100 / n_x)).astype(int), n_x - 1)
    ev["dest_zone_y"] = np.minimum((ev["pos_dest_y"].fillna(-1) // (100 / n_y)).astype(int), n_y - 1)
    ev["orig_zone"] = ev["orig_zone_y"] * n_x + ev["orig_zone_x"]
    ev["dest_zone"] = ev["dest_zone_y"] * n_x + ev["dest_zone_x"]

    # ilerletme yani move ve sut aksiyonlarini belirle
    # ilerletme = isabetli pas + basarili dripling ground attacking duel, won gibi
    is_pass_ok = (ev["eventName"] == "Pass") & (ev["tag_1801"] == 1)
    is_take_on_ok = (ev["subEventName"] == "Ground attacking duel") & (ev["tag_703"] == 1)
    is_move = is_pass_ok | is_take_on_ok
    is_shot = ev["eventName"] == "Shot"
    is_goal = is_shot & (ev["tag_101"] == 1)

    # her bolge icin sut gol hareket sayaclarini cikar
    move_counts = ev.loc[is_move, "orig_zone"].value_counts().reindex(range(n_zones), fill_value=0)
    shot_counts = ev.loc[is_shot, "orig_zone"].value_counts().reindex(range(n_zones), fill_value=0)
    goal_counts = ev.loc[is_goal, "orig_zone"].value_counts().reindex(range(n_zones), fill_value=0)
    total_actions = move_counts + shot_counts

    # o bolgeden sut cekilme olasiligi ile ilerletme yapilma olasiligi ve sutun gol olma olasiligi
    shot_prob = np.where(total_actions > 0, shot_counts / total_actions, 0)
    move_prob = 1 - shot_prob
    goal_prob = np.where(shot_counts > 0, goal_counts / shot_counts, 0)

    # gecis matrisi ile her bolgeden yapilan ilerletmelerin nereye gittigi
    transition_counts = (ev.loc[is_move].groupby(["orig_zone", "dest_zone"]).size()
                         .unstack(fill_value=0).reindex(index=range(n_zones), columns=range(n_zones), fill_value=0))

    row_sums = transition_counts.sum(axis=1).replace(0, np.nan)
    transition_matrix = transition_counts.div(row_sums, axis=0).fillna(0).values

    # her ilerletme aksiyonu icin xt_added hesapla
    # xt_added = bitis bolgesinin degeri - baslangic bolgesinin degeri.
    # sadece "move" aksiyonlari icin anlamli, digerlerinde NaN kalir.
    ev["xt_added"] = np.nan
    ev.loc[is_move, "xt_added"] = xt[ev.loc[is_move, "dest_zone"].values] - xt[ev.loc[is_move, "orig_zone"].values]
    ev.loc[is_move, "xt_added"].mean()
    # oyuncu-mac bazinda topla,
    xt_match = (ev.loc[is_move].groupby(["playerId", "matchId"], as_index=False)["xt_added"].sum()
                .rename(columns={"xt_added": "xt_added_total"}))
    if csv:
        xt_match.to_csv("event_xt.csv", index=False)

    return xt_match

