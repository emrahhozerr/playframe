import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ev = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Feature_Engineer\Data Preprocessing\data\eng_players_data_prep.csv")

# rasgele maç ve takım adisi görselelştirme için
match_id= 2499719
team_id = 1631

# sadece bu mac + bu takimin olaylari, kronolojik sirada
team_ev = ev[(ev["matchId"] == match_id) & (ev["teamId"] == team_id)].copy()
team_ev = team_ev.sort_values(["matchPeriod", "eventSec"]).reset_index(drop=True)
is_pass_ok = (team_ev["eventName"] == "Pass") & (team_ev["tag_1801"] == 1)

# alici tahmini bir sonraki olayin oyuncusu ayni takimin bir sonraki aksiyonu
team_ev["receiver_playerId"] = team_ev["playerId"].shift(-1)

passes = team_ev[is_pass_ok].copy()
passes = passes.dropna(subset=["receiver_playerId"])
passes["receiver_playerId"] = passes["receiver_playerId"].astype(int)

# kendi kendine pas olmaz (teorik olarak olmamali ama guvenlik icin)
passes = passes[passes["playerId"] != passes["receiver_playerId"]]

is_pass_ok.sum()
len(passes)

# oyuncu ortalama pozisyonu kendi orig konumlarinin ortalamasi
avg_pos = team_ev.groupby("playerId")[["pos_orig_x", "pos_orig_y"]].mean()

# oyuncu cifti basina pas sayisi yon onemli degil, karsilikli toplaniyor
pair_counts = {}
for _, row in passes.iterrows():
    a, b = row["playerId"], row["receiver_playerId"]
    key = tuple(sorted([a, b]))
    pair_counts[key] = pair_counts.get(key, 0) + 1

# her oyuncunun toplam pas (attigi+aldigi) sayisi - node buyuklugu icin
pass_volume = passes["playerId"].value_counts().add(passes["receiver_playerId"].value_counts(), fill_value=0)

len(pass_volume)
#en cok pas yapan/alan ilk 5 oyuncu
pass_volume.sort_values(ascending=False).head()

# gorsellestirme sol pas agi  sag bolge yogunluk ısı haritasi

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8))

for ax in (ax1, ax2):
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.add_patch(plt.Rectangle((0, 0), 105, 68, fill=False, color="black"))
    ax.axvline(52.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_aspect("equal")

# sol: pas agi
for (a, b), count in pair_counts.items():
    if a not in avg_pos.index or b not in avg_pos.index:
        continue
    x1, y1 = avg_pos.loc[a, "pos_orig_x"] / 100 * 105, avg_pos.loc[a, "pos_orig_y"] / 100 * 68
    x2, y2 = avg_pos.loc[b, "pos_orig_x"] / 100 * 105, avg_pos.loc[b, "pos_orig_y"] / 100 * 68
    ax1.plot([x1, x2], [y1, y2], color="steelblue", alpha=0.5, linewidth=count / 3)

for player_id, vol in pass_volume.items():
    if player_id not in avg_pos.index:
        continue
    x, y = avg_pos.loc[player_id, "pos_orig_x"] / 100 * 105, avg_pos.loc[player_id, "pos_orig_y"] / 100 * 68
    ax1.scatter(x, y, s=vol * 15, color="firebrick", edgecolor="black", zorder=3)
    ax1.text(x, y + 2, str(int(player_id)), ha="center", fontsize=8, zorder=4)

ax1.set_title(f"Pas Agi - matchId={match_id}, teamId={team_id}")

#sag butun aksiyonlarin bolge yogunlugu ısı haritasi
x_m = team_ev["pos_orig_x"] / 100 * 105
y_m = team_ev["pos_orig_y"] / 100 * 68
hb = ax2.hexbin(x_m, y_m, gridsize=18, extent=(0, 105, 0, 68), cmap="Reds", mincnt=1)
fig.colorbar(hb, ax=ax2, label="aksiyon sayisi")
ax2.set_title(f"Bolge Yogunlugu - matchId={match_id}, teamId={team_id}")

plt.tight_layout()
plt.savefig("pass_network.png", dpi=150)


def pass_network(ev, match_id, team_id):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    ev = ev

    # rasgele maç ve takım adisi görselelştirme için
    match_id = match_id
    team_id = team_id

    # sadece bu mac + bu takimin olaylari, kronolojik sirada
    team_ev = ev[(ev["matchId"] == match_id) & (ev["teamId"] == team_id)].copy()
    team_ev = team_ev.sort_values(["matchPeriod", "eventSec"]).reset_index(drop=True)
    is_pass_ok = (team_ev["eventName"] == "Pass") & (team_ev["tag_1801"] == 1)

    # alici tahmini bir sonraki olayin oyuncusu ayni takimin bir sonraki aksiyonu
    team_ev["receiver_playerId"] = team_ev["playerId"].shift(-1)

    passes = team_ev[is_pass_ok].copy()
    passes = passes.dropna(subset=["receiver_playerId"])
    passes["receiver_playerId"] = passes["receiver_playerId"].astype(int)

    # kendi kendine pas olmaz (teorik olarak olmamali ama guvenlik icin)
    passes = passes[passes["playerId"] != passes["receiver_playerId"]]

    # oyuncu ortalama pozisyonu kendi orig konumlarinin ortalamasi
    avg_pos = team_ev.groupby("playerId")[["pos_orig_x", "pos_orig_y"]].mean()

    # oyuncu cifti basina pas sayisi yon onemli degil, karsilikli toplaniyor
    pair_counts = {}
    for _, row in passes.iterrows():
        a, b = row["playerId"], row["receiver_playerId"]
        key = tuple(sorted([a, b]))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    # her oyuncunun toplam pas (attigi+aldigi) sayisi - node buyuklugu icin
    pass_volume = passes["playerId"].value_counts().add(passes["receiver_playerId"].value_counts(), fill_value=0)

    # gorsellestirme sol pas agi  sag bolge yogunluk ısı haritasi
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8))

    for ax in (ax1, ax2):
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 68)
        ax.add_patch(plt.Rectangle((0, 0), 105, 68, fill=False, color="black"))
        ax.axvline(52.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_aspect("equal")

    # sol: pas agi
    for (a, b), count in pair_counts.items():
        if a not in avg_pos.index or b not in avg_pos.index:
            continue
        x1, y1 = avg_pos.loc[a, "pos_orig_x"] / 100 * 105, avg_pos.loc[a, "pos_orig_y"] / 100 * 68
        x2, y2 = avg_pos.loc[b, "pos_orig_x"] / 100 * 105, avg_pos.loc[b, "pos_orig_y"] / 100 * 68
        ax1.plot([x1, x2], [y1, y2], color="steelblue", alpha=0.5, linewidth=count / 3)

    for player_id, vol in pass_volume.items():
        if player_id not in avg_pos.index:
            continue
        x, y = avg_pos.loc[player_id, "pos_orig_x"] / 100 * 105, avg_pos.loc[player_id, "pos_orig_y"] / 100 * 68
        ax1.scatter(x, y, s=vol * 15, color="firebrick", edgecolor="black", zorder=3)
        ax1.text(x, y + 2, str(int(player_id)), ha="center", fontsize=8, zorder=4)

    ax1.set_title(f"Pas Agi - matchId={match_id}, teamId={team_id}")

    # sag butun aksiyonlarin bolge yogunlugu ısı haritasi
    x_m = team_ev["pos_orig_x"] / 100 * 105
    y_m = team_ev["pos_orig_y"] / 100 * 68
    hb = ax2.hexbin(x_m, y_m, gridsize=18, extent=(0, 105, 0, 68), cmap="Reds", mincnt=1)
    fig.colorbar(hb, ax=ax2, label="aksiyon sayisi")
    ax2.set_title(f"Bolge Yogunlugu - matchId={match_id}, teamId={team_id}")

    plt.tight_layout()
    plt.savefig("pass_network.png", dpi=150)
    plt.close(fig)
