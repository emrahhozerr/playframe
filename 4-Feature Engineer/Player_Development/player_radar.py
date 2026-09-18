import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

norm = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Feature_Engineer\Player_Features\Data\player_season_normalized.csv")


# pozisyona ozel kurasyon edilmis metrik setleri
# her pozisyon icin o pozisyonu anlamli sekilde tanimlayan 6-7 metrik seciyoruz.
# 90 metrigin hepsini radara koymak okunmaz olurdu bu yüzden StatsBomb,FBref,tarzi
# kart mantigi bu pozisyona gore az sayida, anlamli eksen
radar_metrics = {
    "FW": {
        "goals_p90": "Gol",
        "shots_on_target_p90": "Isabetli Sut",
        "xt_added_total_p90": "xT",
        "key_passes_p90": "Kilit Pas",
        "take_ons_won_p90": "Basarili Calim",
        "assists_p90": "Asist",
        "aerial_duels_won_p90": "Hava Toplari",
    },
    "MD": {
        "passes_accurate_p90": "Isabetli Pas",
        "key_passes_p90": "Kilit Pas",
        "through_passes_p90": "Kesme Pas",
        "xt_added_total_p90": "xT",
        "interceptions_p90": "Top Kapma",
        "duels_won_p90": "Kazanilan Duello",
        "take_ons_won_p90": "Basarili Calim",
    },
    "DF": {
        "duels_won_p90": "Kazanilan Duello",
        "aerial_duels_won_p90": "Hava Toplari",
        "interceptions_p90": "Top Kapma",
        "clearances_p90": "Uzaklastirma",
        "sliding_tackles_p90": "Kayarak Mudahale",
        "passes_accurate_p90": "Isabetli Pas",
        "blocks_p90": "Blok",
    },
    "GK": {
        "reflex_saves_p90": "Refleks Kurtaris",
        "save_attempts_p90": "Kurtaris Denemesi",
        "passes_accurate_p90": "Isabetli Pas",
        "long_passes_p90": "Uzun Pas",
        "gk_leaving_line_p90": "Cizgi Disi Mudahale",
        "goals_conceded_event_p90": "Az Gol Yeme",  # ters metrik, asagida invert ediliyor
    },
}

# bu metriklerde dusuk ham deger iyidir az gol yemek gibi - radarda disari
# iyi okunabilsin diye percentile'i 100'den cikarip terse ceviriyoruz
invert_metrics = {"goals_conceded_event_p90"}


def get_percentiles(player_name):
    row = norm[norm["player_name"] == player_name]
    if row.empty:
        print("oyuncu bulunamadi:", player_name)
        return None, None
    row = row.iloc[0]
    role = row["role_code"]
    metrics = radar_metrics[role]

    values = []
    for metric in metrics:
        pct_col = f"{metric}_percentile"
        pct = row[pct_col]
        if metric in invert_metrics and pd.notnull(pct):
            pct = 100 - pct
        values.append(pct if pd.notnull(pct) else 0)
    return metrics, values


def plot_radar(player_names):
    if isinstance(player_names, str):
        player_names = [player_names]

    # ilk oyuncunun pozisyonundan eksen etiketlerini al karsilastirmada
    # ayni pozisyon grubundan oyuncu secilmesi bekleniyor
    metrics, _ = get_percentiles(player_names[0])
    if metrics is None:
        return
    labels = list(metrics.values())
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8.5), subplot_kw=dict(polar=True))
    colors = ["firebrick", "steelblue", "seagreen"]

    for i, player_name in enumerate(player_names):
        metrics_i, values = get_percentiles(player_name)
        if metrics_i is None:
            continue
        values = values + values[:1]
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=player_name)
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    fig.suptitle(" vs ".join(player_names) + "\npercentile radar (pozisyon grubu icinde)", fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    safe_name = "_".join(player_names).replace(" ", "_")
    out_path = f"player_radar_{safe_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("kaydedildi:", out_path)



# genis tablo pozisyona gore kurasyon yapmadan TUM p90 metriklerinin
# percentile'ini gosteren yatay bar - radar degil ama tam istatistik ozeti

def full_stat_sheet(player_name, top_n=None):
    row = norm[norm["player_name"] == player_name]
    if row.empty:
        print("oyuncu bulunamadi:", player_name)
        return None
    row = row.iloc[0]

    pct_cols = [c for c in norm.columns if c.endswith("_p90_percentile")]
    data = row[pct_cols].dropna().astype(float)
    data.index = [c.replace("_p90_percentile", "") for c in data.index]
    data = data.sort_values(ascending=True)
    if top_n:
        data = data.tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(6, len(data) * 0.22)))
    colors = plt.cm.RdYlGn(data.values / 100)
    ax.barh(data.index, data.values, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile (pozisyon grubu icinde)")
    ax.set_title(f"{player_name} - Tam Istatistik Ozeti")
    plt.tight_layout()
    plt.savefig("player_full_stats.png", dpi=150)
    plt.close(fig)
    print("kaydedildi: player_full_stats.png")


plot_radar("Eden Hazard")
plot_radar(["Eden Hazard", "Alexis Alejandro Sánchez Sánchez"])
full_stat_sheet("Eden Hazard", top_n=25)

####################################################################
# Foksiyon
def player_radar_func(norm, player_names, top_n=25):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    norm = norm
    # pozisyona ozel kurasyon edilmis metrik setleri
    # her pozisyon icin o pozisyonu anlamli sekilde tanimlayan 6-7 metrik seciyoruz.
    # 90 metrigin hepsini radara koymak okunmaz olurdu - StatsBomb/FBref tarzi
    # kart mantigi bu (pozisyona gore az sayida, anlamli eksen)
    radar_metrics = {
        "FW": {
            "goals_p90": "Gol",
            "shots_on_target_p90": "Isabetli Sut",
            "xt_added_total_p90": "xT",
            "key_passes_p90": "Kilit Pas",
            "take_ons_won_p90": "Basarili Calim",
            "assists_p90": "Asist",
            "aerial_duels_won_p90": "Hava Toplari",
        },
        "MD": {
            "passes_accurate_p90": "Isabetli Pas",
            "key_passes_p90": "Kilit Pas",
            "through_passes_p90": "Kesme Pas",
            "xt_added_total_p90": "xT",
            "interceptions_p90": "Top Kapma",
            "duels_won_p90": "Kazanilan Duello",
            "take_ons_won_p90": "Basarili Calim",
        },
        "DF": {
            "duels_won_p90": "Kazanilan Duello",
            "aerial_duels_won_p90": "Hava Toplari",
            "interceptions_p90": "Top Kapma",
            "clearances_p90": "Uzaklastirma",
            "sliding_tackles_p90": "Kayarak Mudahale",
            "passes_accurate_p90": "Isabetli Pas",
            "blocks_p90": "Blok",
        },
        "GK": {
            "reflex_saves_p90": "Refleks Kurtaris",
            "save_attempts_p90": "Kurtaris Denemesi",
            "passes_accurate_p90": "Isabetli Pas",
            "long_passes_p90": "Uzun Pas",
            "gk_leaving_line_p90": "Cizgi Disi Mudahale",
            "goals_conceded_event_p90": "Az Gol Yeme",  # ters metrik, asagida invert ediliyor
        },
    }

    # bu metriklerde dusuk ham deger iyidir (az gol yemek gibi) - radarda "disari =
    # iyi" okunabilsin diye percentile'i 100'den cikarip terse ceviriyoruz
    invert_metrics = {"goals_conceded_event_p90"}

    def get_percentiles(player_name):
        row = norm[norm["player_name"] == player_name]
        if row.empty:
            print("oyuncu bulunamadi:", player_name)
            return None, None
        row = row.iloc[0]
        role = row["role_code"]
        metrics = radar_metrics[role]

        values = []
        for metric in metrics:
            pct_col = f"{metric}_percentile"
            pct = row[pct_col]
            if metric in invert_metrics and pd.notnull(pct):
                pct = 100 - pct
            values.append(pct if pd.notnull(pct) else 0)
        return metrics, values

    def plot_radar(names):
        metrics, _ = get_percentiles(names[0])
        if metrics is None:
            return None
        labels = list(metrics.values())
        n = len(labels)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8.5), subplot_kw=dict(polar=True))
        colors = ["firebrick", "steelblue", "seagreen"]

        for i, player_name in enumerate(names):
            metrics_i, values = get_percentiles(player_name)
            if metrics_i is None:
                continue
            values = values + values[:1]
            ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=player_name)
            ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        fig.suptitle(" vs ".join(names) + "\npercentile radar (pozisyon grubu icinde)", fontsize=11)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        safe_name = "_".join(names).replace(" ", "_")
        out_path = f"player_radar_{safe_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print("kaydedildi:", out_path)
        return out_path

    def full_stat_sheet(player_name, n):
        row = norm[norm["player_name"] == player_name]
        if row.empty:
            print("oyuncu bulunamadi:", player_name)
            return None
        row = row.iloc[0]

        pct_cols = [c for c in norm.columns if c.endswith("_p90_percentile")]
        data = row[pct_cols].dropna().astype(float)
        data.index = [c.replace("_p90_percentile", "") for c in data.index]
        data = data.sort_values(ascending=True)
        if n:
            data = data.tail(n)

        fig, ax = plt.subplots(figsize=(8, max(6, len(data) * 0.22)))
        colors = plt.cm.RdYlGn(data.values / 100)
        ax.barh(data.index, data.values, color=colors)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentile (pozisyon grubu icinde)")
        ax.set_title(f"{player_name} - Tam Istatistik Ozeti")
        plt.tight_layout()
        out_path = f"player_full_stats_{player_name.replace(' ', '_')}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print("kaydedildi:", out_path)
        return out_path


    # tek isim de gelse, birden fazla isim de gelse calissin
    if isinstance(player_names, str):
        player_names = [player_names]

    radar_path = plot_radar(player_names)

    # tam istatistik ozeti sadece tek oyuncu icin anlamli (karsilastirmada degil)
    stat_path = None
    if len(player_names) == 1:
        stat_path = full_stat_sheet(player_names[0], top_n)

    return radar_path, stat_path


import pandas as pd
norm = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Feature_Engineer\Player_Features\Data\player_season_normalized.csv")

player_radar_func(norm,"Eden Hazard")
player_radar_func(norm,["Eden Hazard", "Alexis Sánchez"])