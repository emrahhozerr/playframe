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

def perc90_builder(event_eng,player_game,csv=False):
    import pandas as pd
    import numpy as np

    # tag sozlugu - sabit tanimlar
    # sut bolgesi tag'leri 3 gruba ayriliyor (kaleye isabetli / auta / direk).
    # kaleye isabetli
    on_target_tags = [f"tag_{col}" for col in range(1201, 1210)]
    # auta
    off_target_tags = [f"tag_{col}" for col in range(1210, 1217)]
    # direk
    post_tags = [f"tag_{col}" for col in range(1217, 1224)]
    # tümünü birleştir
    all_zone_tags = on_target_tags + off_target_tags + post_tags

    zone_labels = {
        "tag_1201": "zone_goal_low_center", "tag_1202": "zone_goal_low_right",
        "tag_1203": "zone_goal_center", "tag_1204": "zone_goal_center_left",
        "tag_1205": "zone_goal_low_left", "tag_1206": "zone_goal_center_right",
        "tag_1207": "zone_goal_high_center", "tag_1208": "zone_goal_high_left",
        "tag_1209": "zone_goal_high_right",
        "tag_1210": "zone_out_low_right", "tag_1211": "zone_out_center_left",
        "tag_1212": "zone_out_low_left", "tag_1213": "zone_out_center_right",
        "tag_1214": "zone_out_high_center", "tag_1215": "zone_out_high_left",
        "tag_1216": "zone_out_high_right",
        "tag_1217": "zone_post_low_right", "tag_1218": "zone_post_center_left",
        "tag_1219": "zone_post_low_left", "tag_1220": "zone_post_center_right",
        "tag_1221": "zone_post_high_center", "tag_1222": "zone_post_high_left",
        "tag_1223": "zone_post_high_right",
    }

    all_tags = (
            ["tag_101", "tag_102", "tag_201", "tag_301", "tag_302",
             "tag_401", "tag_402", "tag_403", "tag_501", "tag_502", "tag_503", "tag_504",
             "tag_601", "tag_602", "tag_701", "tag_702", "tag_703",
             "tag_801", "tag_901", "tag_1001", "tag_1101", "tag_1102"]
            + all_zone_tags
            + ["tag_1301", "tag_1302", "tag_1401", "tag_1601",
               "tag_1701", "tag_1702", "tag_1703", "tag_1801", "tag_1802",
               "tag_1901", "tag_2001", "tag_2101"]
    )

    # ham event verisini isle
    # event_eng zaten yukarida tum kolonlariyla okundu burada sadece ihtiyacimiz
    # olan kolonlari secip dtype'lari optimize ediyoruz
    base_cols = ["playerId", "matchId", "teamId", "eventName", "subEventName"]
    usecols = base_cols + all_tags

    dtype_map = {c: "int8" for c in all_tags}
    dtype_map.update({
        "playerId": "int32", "matchId": "int32", "teamId": "int32",
        "eventName": "category", "subEventName": "category",
    })

    ev = event_eng[usecols].astype(dtype_map)

    # boolean maskeler - hangi satir hangi tur aksiyon?
    is_pass = ev["eventName"] == "Pass"
    is_duel = ev["eventName"] == "Duel"
    is_shot = ev["eventName"].isin(["Shot"]) | (ev["subEventName"].isin(["Free kick shot", "Penalty"]))
    is_foul = ev["eventName"] == "Foul"
    is_offside = ev["eventName"] == "Offside"
    is_free_kick = ev["eventName"] == "Free Kick"

    sub = ev["subEventName"]
    is_cross = sub.isin(["Cross", "Free kick cross"])
    is_long_pass = sub.isin(["Launch", "High pass"])
    is_smart_pass = sub == "Smart pass"
    is_air_duel = sub == "Air duel"
    is_ground_def_duel = sub == "Ground defending duel"
    is_ground_att_duel = sub == "Ground attacking duel"
    is_loose_ball_duel = sub == "Ground loose ball duel"
    is_clearance = sub == "Clearance"
    is_touch = sub == "Touch"
    is_acceleration = sub == "Acceleration"
    is_corner = sub == "Corner"
    is_throw_in = sub == "Throw in"
    is_goal_kick = sub == "Goal kick"
    is_save_attempt = ev["eventName"] == "Save attempt"
    is_reflexes = sub == "Reflexes"
    is_gk_leaving_line = ev["eventName"] == "Goalkeeper leaving line"

    is_take_on = (ev["tag_503"] == 1) | (ev["tag_504"] == 1)
    is_on_target = ev[on_target_tags].sum(axis=1) > 0
    is_off_target = ev[off_target_tags].sum(axis=1) > 0
    is_post = ev[post_tags].sum(axis=1) > 0

    # feature tablosu - her satir icin 0/1 kolonlar
    feat = pd.DataFrame({
        "playerId": ev["playerId"],
        "matchId": ev["matchId"],
        "teamId": ev["teamId"],

        # --- pas ---
        "passes_total": is_pass.astype("int8"),
        "passes_accurate": (is_pass & (ev["tag_1801"] == 1)).astype("int8"),
        "key_passes": ev["tag_302"],
        "assists": ev["tag_301"],
        "opportunities_created": ev["tag_201"],
        "through_passes": (is_pass & (ev["tag_901"] == 1)).astype("int8"),
        "crosses": is_cross.astype("int8"),
        "long_passes": is_long_pass.astype("int8"),
        "smart_passes": is_smart_pass.astype("int8"),
        "high_ball_actions": ev["tag_801"],
        "free_kick_direct": (is_free_kick & (ev["tag_1101"] == 1)).astype("int8"),
        "free_kick_indirect": (is_free_kick & (ev["tag_1102"] == 1)).astype("int8"),

        # --- ayak / vucut tercihi ---
        "left_foot_actions": ev["tag_401"],
        "right_foot_actions": ev["tag_402"],
        "head_body_actions": ev["tag_403"],

        # --- hareket / bosluk ---
        "free_space_right_actions": ev["tag_501"],
        "free_space_left_actions": ev["tag_502"],

        # --- ikili mucadele (won + lost + neutral = total) ---
        "duels_total": is_duel.astype("int8"),
        "duels_won": (is_duel & (ev["tag_703"] == 1)).astype("int8"),
        "duels_lost": (is_duel & (ev["tag_701"] == 1)).astype("int8"),
        "duels_neutral": (is_duel & (ev["tag_702"] == 1)).astype("int8"),
        "aerial_duels": is_air_duel.astype("int8"),
        "aerial_duels_won": (is_air_duel & (ev["tag_703"] == 1)).astype("int8"),
        "ground_def_duels": is_ground_def_duel.astype("int8"),
        "ground_def_duels_won": (is_ground_def_duel & (ev["tag_703"] == 1)).astype("int8"),
        "ground_att_duels": is_ground_att_duel.astype("int8"),
        "ground_att_duels_won": (is_ground_att_duel & (ev["tag_703"] == 1)).astype("int8"),
        "loose_ball_duels": is_loose_ball_duel.astype("int8"),
        "loose_ball_duels_won": (is_loose_ball_duel & (ev["tag_703"] == 1)).astype("int8"),

        # --- cift / dribbling ---
        "take_ons": is_take_on.astype("int8"),
        "take_ons_won": (is_take_on & (ev["tag_703"] == 1)).astype("int8"),
        "feints": ev["tag_1301"],

        # --- savunma ---
        "interceptions": ev["tag_1401"],
        "sliding_tackles": ev["tag_1601"],
        "clearances": is_clearance.astype("int8"),
        "blocks": ev["tag_2101"],
        "dangerous_ball_lost": ev["tag_2001"],
        "anticipated_against": ev["tag_601"],
        "anticipation_actions": ev["tag_602"],

        # --- hata ---
        "missed_ball": ev["tag_1302"],
        "own_goals": ev["tag_102"],

        # --- disiplin ---
        "fouls_committed": is_foul.astype("int8"),
        "yellow_cards": ev["tag_1702"],
        "second_yellow_cards": ev["tag_1703"],
        "red_cards_direct": ev["tag_1701"],
        "red_cards": (ev["tag_1701"] + ev["tag_1703"]).clip(upper=1).astype("int8"),
        "fairplay_actions": ev["tag_1001"],

        # --- sut / gol ---
        # dikkat: tag_101 sadece sut degil, kalecinin "save attempt" eventine
        # de (golu kurtaramayinca) etiketleniyor. is_shot ile kisitlamazsak
        # kaleciler "en golcu oyuncular" gibi gorunur (bu hata ilk denemede
        # yakalanmisti, buraya not dusuyorum).
        "shots_total": is_shot.astype("int8"),
        "goals": (is_shot & (ev["tag_101"] == 1)).astype("int8"),
        "goals_conceded_event": ((ev["eventName"] == "Save attempt") & (ev["tag_101"] == 1)).astype("int8"),
        "shots_on_target": (is_shot & is_on_target).astype("int8"),
        "shots_off_target": (is_shot & is_off_target).astype("int8"),
        "shots_post": (is_shot & is_post).astype("int8"),
        "shots_blocked": (is_shot & (ev["tag_2101"] == 1)).astype("int8"),

        # --- top kontrolu / diger ---
        "touches": is_touch.astype("int8"),
        "accelerations": is_acceleration.astype("int8"),
        "counter_attack_actions": ev["tag_1901"],
        "offsides": is_offside.astype("int8"),
        "accurate_actions_total": ev["tag_1801"],
        "inaccurate_actions_total": ev["tag_1802"],

        # --- kaleci ---
        "save_attempts": is_save_attempt.astype("int8"),
        "reflex_saves": is_reflexes.astype("int8"),
        "gk_leaving_line": is_gk_leaving_line.astype("int8"),

        # --- duran top ---
        "corners_taken": is_corner.astype("int8"),
        "throw_ins": is_throw_in.astype("int8"),
        "goal_kicks": is_goal_kick.astype("int8"),
    })

    # sut bolgesi ayrı columns oluşturma
    zone_feat = {}
    for tag_col, label in zone_labels.items():
        zone_feat[label] = (is_shot & (ev[tag_col] == 1)).astype("int8")
    feat = pd.concat([feat, pd.DataFrame(zone_feat)], axis=1)

    # artik ham tabloya ihtiyac yok, bellekten silelim
    del ev

    # groupby - oyuncu-mac bazinda toplama  birleşitrme öncesi
    count_cols = [col for col in feat.columns if col not in ("playerId", "matchId", "teamId")]
    agg = feat.groupby(["playerId", "matchId", "teamId"], as_index=False)[count_cols].sum()

    # player_game (dakika + meta) verisini işle
    # player_game zaten yukarida tum kolonlariyla okundu, burada ihtiyacimiz
    # olan kolonlari seciyoruz

    pg_cols = [
        "game_id", "player_id", "team_id", "player_name", "is_starter",
        "minutes_played", "player_info_role_code", "player_info_role_name",
        "match_gameweek",
        "birth_date", "player_info_height", "player_info_weight",
        "player_info_foot", "player_info_birth_country",
        "player_info_passport_country", "player_info_currentTeamId",
        "team_team_name", "team_city", "team_area_team_conutry",
        "match_dateutc", "match_venue",
        "match_team1.teamId", "match_team1.side", "match_team1.score", "match_team1.scoreHT",
        "match_team2.teamId", "match_team2.side", "match_team2.score", "match_team2.scoreHT",
    ]
    pg = player_game[pg_cols].copy()
    pg = pg.rename(columns={"game_id": "matchId", "player_id": "playerId", "team_id": "teamId"})

    # oyuncu yasi hesapla
    pg["match_dateutc"] = pd.to_datetime(pg["match_dateutc"])
    pg["birth_date"] = pd.to_datetime(pg["birth_date"])
    pg["player_age_at_match"] = ((pg["match_dateutc"] - pg["birth_date"]).dt.days / 365.25).round(1)
    pg = pg.drop(columns=["birth_date"])

    # match_team1,match_team2 ev sahibi takımım ve rakip takım gibi metriklere çevriliyor karışıklığı önlemek için
    is_team1 = pg["teamId"] == pg["match_team1.teamId"]

    pg["opponent_team_id"] = np.where(is_team1, pg["match_team2.teamId"], pg["match_team1.teamId"])
    pg["is_home"] = np.where(is_team1, pg["match_team1.side"] == "home", pg["match_team2.side"] == "home")
    pg["team_goals"] = np.where(is_team1, pg["match_team1.score"], pg["match_team2.score"])
    pg["opponent_goals"] = np.where(is_team1, pg["match_team2.score"], pg["match_team1.score"])
    pg["team_goals_HT"] = np.where(is_team1, pg["match_team1.scoreHT"], pg["match_team2.scoreHT"])
    pg["opponent_goals_HT"] = np.where(is_team1, pg["match_team2.scoreHT"], pg["match_team1.scoreHT"])
    pg["match_result"] = np.select(
        [pg["team_goals"] > pg["opponent_goals"], pg["team_goals"] < pg["opponent_goals"]],
        ["W", "L"], default="D",
    )

    pg = pg.drop(columns=[
        "match_team1.teamId", "match_team1.side", "match_team1.score", "match_team1.scoreHT",
        "match_team2.teamId", "match_team2.side", "match_team2.score", "match_team2.scoreHT",
    ])

    # iki tabloyu birlestir (merge)
    merged = pg.merge(agg, on=["playerId", "matchId", "teamId"], how="left")
    merged[count_cols] = merged[count_cols].fillna(0)

    # per-90 hesapla
    minutes = merged["minutes_played"].replace(0, np.nan)
    per90 = merged[count_cols].div(minutes, axis=0) * 90
    per90.columns = [f"{col}_p90" for col in count_cols]

    # oran (%) metrikleri hesapla
    merged["pass_accuracy_pct"] = np.where(
        merged["passes_total"] > 0, merged["passes_accurate"] / merged["passes_total"] * 100, np.nan
    )
    merged["overall_accuracy_pct"] = np.where(
        (merged["accurate_actions_total"] + merged["inaccurate_actions_total"]) > 0,
        merged["accurate_actions_total"] / (
                    merged["accurate_actions_total"] + merged["inaccurate_actions_total"]) * 100,
        np.nan,
    )
    merged["duel_win_pct"] = np.where(merged["duels_total"] > 0, merged["duels_won"] / merged["duels_total"] * 100,
                                      np.nan)
    merged["duel_loss_pct"] = np.where(merged["duels_total"] > 0, merged["duels_lost"] / merged["duels_total"] * 100,
                                       np.nan)
    merged["duel_neutral_pct"] = np.where(merged["duels_total"] > 0,
                                          merged["duels_neutral"] / merged["duels_total"] * 100, np.nan)
    merged["take_on_success_pct"] = np.where(merged["take_ons"] > 0, merged["take_ons_won"] / merged["take_ons"] * 100,
                                             np.nan)
    merged["shot_accuracy_pct"] = np.where(merged["shots_total"] > 0,
                                           merged["shots_on_target"] / merged["shots_total"] * 100, np.nan)
    merged["shot_conversion_pct"] = np.where(merged["shots_total"] > 0, merged["goals"] / merged["shots_total"] * 100,
                                             np.nan)
    foot_total = merged["left_foot_actions"] + merged["right_foot_actions"]
    merged["right_foot_dominance_pct"] = np.where(foot_total > 0, merged["right_foot_actions"] / foot_total * 100,
                                                  np.nan)

    # ham + per90 + oran tek tabloda birlestir
    result = pd.concat([merged, per90], axis=1)

    if csv:
        result.to_csv("perc90_builder.csv", index=False)
    return result

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

def scout_engine_func(norm,user,head=10):
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans

    norm = norm

    # feature seçimi sadece z-score kolonlarini kullaniyoruz
    # z-score'lar zaten pozisyon grubuna role_code gore hesaplandigi icin
    # farkli olcekteki metrikler orn. pas sayisi vs gol sayisi esit agirlikta
    # kiyaslanabiliyor. yuzde/percentile/ham kolonlari feature'a katmiyoruz -
    # ayni bilginin farkli bir gosterimi, cift sayim olur.
    feature_cols = [col for col in norm.columns if col.endswith("_p90_z")]

    # std=0 olan gruplarda ya da o metrik o pozisyonda hic olmayan durumlarda
    # z-score NaN kalabilir - 0 grubun ortalamasi ile dolduruyoruz ki o metrik
    # o oyuncu icin "notr" sayilsin, benzerlik hesabini bozmasin
    X_all = norm[feature_cols].fillna(0)

    # aday havuzu sadece esik ustu 600 dakika oyuncular seçilcek
    # az dakikali oyuncularin z-skoru gürültülü olabilir kucuk ornekte asiri
    # uc degerler onerilecek/kiyaslanacak havuzu guvenilir oyuncularla
    # sinirliyoruz. aranan oyuncu esik altinda olsa bile onun icin sorgu
    # yapilabilir, sadece kiyaslandigi havuz guvenilir oyunculardan olusuyor
    pool = norm[norm["is_eligible"]].copy()
    pool_X = X_all.loc[pool.index]

    # pozisyon kisitli k-means her pozisyon grubunda ayri oyun stili kumesi
    # z-skorlar pozisyona gore hesaplandigi icin farkli pozisyonlari ayni
    # k-means'e sokmak anlamsiz olur bir kalecinin hucum z-skoru zaten
    # baska bir olcekte o yuzden gruplandirmayi role_code bazinda ayri ayri yapiyoruz
    n_clusters = 4
    pool["style_cluster"] = -1

    for role, idx in pool.groupby("role_code").groups.items():
        X_role = pool_X.loc[idx]
        if len(X_role) < n_clusters:
            continue
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pool.loc[idx, "style_cluster"] = km.fit_predict(X_role)


    # cosine similarity: ayni pozisyon grubu icinde en benzer oyunculari bul
    def similar_players(player_name, top_n=10):
        matched = norm[norm["player_name"] == player_name]
        if matched.empty:
            print("player not found:", player_name)
            return None

        target = matched.iloc[0]
        role = target["role_code"]
        target_vector = X_all.loc[[target.name]]

        candidate_pool = pool[pool["role_code"] == role]
        candidate_vectors = pool_X.loc[candidate_pool.index]

        similarities = cosine_similarity(target_vector, candidate_vectors)[0]

        result = candidate_pool[["player_name", "role_code", "style_cluster", "minutes_played"]].copy()
        result["similarity"] = similarities
        result = result[result["player_name"] != player_name]
        result = result.sort_values("similarity", ascending=False).head(top_n)
        return result

    scout_user = similar_players(user, top_n=head).sort_values("similarity",ascending=False)
    # kume etiketini pool'dan norm'a geri tasi ki kaydedilen dosyada da olsun
    norm = norm.merge(pool[["playerId", "style_cluster"]], on="playerId", how="left")
    norm.to_csv("player_season_scouted.csv", index=False)
    return scout_user

def tactic_builder_func(ev):
    import pandas as pd
    import numpy as np
    import warnings

    ev = pd.read_csv(ev)
    # saha olcekleri pitch 0-100 grid, gercek metreye ceviriyoruz
    length_m = 105
    width_m = 68

    dx_m = (ev["pos_dest_x"] - ev["pos_orig_x"]) / 100 * length_m
    dy_m = (ev["pos_dest_y"] - ev["pos_orig_y"]) / 100 * width_m
    ev["pass_distance_m"] = np.sqrt(dx_m ** 2 + dy_m ** 2)
    ev["forward_progress"] = ev["pos_dest_x"] - ev["pos_orig_x"]

    # olay bayraklari
    is_pass = ev["eventName"] == "Pass"
    is_pass_ok = is_pass & ev["tag_1801"]
    is_long_ball = is_pass & (ev["pass_distance_m"] >= 30)

    # son 1/3'e giris baslangic son 1/3 disinda, bitis son 1/3 icinde
    is_final_third_entry = is_pass & (ev["pos_orig_x"] < 66) & (ev["pos_dest_x"] >= 66)

    # ceza sahasi girisi bitis noktasi kabaca ceza sahasi dikdortgeninde
    is_box_entry = is_pass & (ev["pos_dest_x"] >= 84) & (ev["pos_dest_y"].between(20, 80))

    # kanat aksiyonu: baslangic noktasi genislik kanatlarinda
    is_wide_action = is_pass & ((ev["pos_orig_y"] < 33) | (ev["pos_orig_y"] > 66))

    # pressing savunma duellosu ya da faul, rakip yari sahada kendi hucum yonune gore x>=60
    # not= duel olaylari iki takim icin de ayri satir olarak, birbirinin aynasi 100-x,100-y
    # koordinatla kayitli - yani orig_x>=60 olan bir savunma duellosu o takimin kendi
    # hucum yonune gore rakip yari sahada oldugu anlamina gelir = yuksek pressing
    is_defensive_duel = ev["subEventName"] == "Ground defending duel"
    is_foul = ev["eventName"] == "Foul"
    is_high_press_action = (is_defensive_duel | is_foul) & (ev["pos_orig_x"] >= 60)

    # set piece: Free Kick eventName'i corner/taca/frikik/penalti/kale vurusunu kapsiyor
    is_set_piece = ev["eventName"] == "Free Kick"
    is_shot = ev["eventName"] == "Shot"
    is_goal = is_shot & ev["tag_101"]

    # takim-mac bazinda topla
    tmp = pd.DataFrame({
        "matchId": ev["matchId"],
        "teamId": ev["teamId"],
        "is_pass": is_pass,
        "is_pass_ok": is_pass_ok,
        "is_long_ball": is_long_ball,
        "is_final_third_entry": is_final_third_entry,
        "is_box_entry": is_box_entry,
        "is_wide_action": is_wide_action,
        "is_defensive_duel": is_defensive_duel,
        "is_high_press_action": is_high_press_action,
        "is_set_piece": is_set_piece,
        "is_shot": is_shot,
        "is_goal": is_goal,
        "pass_distance_m": ev["pass_distance_m"],
        "forward_progress": ev["forward_progress"]})

    team_match = tmp.groupby(["matchId", "teamId"], as_index=False).agg(
        passes_total=("is_pass", "sum"),
        passes_accurate=("is_pass_ok", "sum"),
        long_balls=("is_long_ball", "sum"),
        final_third_entries=("is_final_third_entry", "sum"),
        box_entries=("is_box_entry", "sum"),
        wide_actions=("is_wide_action", "sum"),
        defensive_duels=("is_defensive_duel", "sum"),
        high_press_actions=("is_high_press_action", "sum"),
        set_pieces=("is_set_piece", "sum"),
        shots_total=("is_shot", "sum"),
        goals=("is_goal", "sum"),
        avg_pass_distance_m=("pass_distance_m", "mean"),
        avg_forward_progress=("forward_progress", "mean"))

    # oranlar - ham sayaclarin dogrudan kiyaslanmasi yerine oyunun temposuna gore normalize ediyoruz
    team_match["pass_accuracy_pct"] = np.where(team_match["passes_total"] > 0,
                                               team_match["passes_accurate"] / team_match["passes_total"] * 100, np.nan)
    team_match["long_ball_pct"] = np.where(team_match["passes_total"] > 0,
                                           team_match["long_balls"] / team_match["passes_total"] * 100, np.nan)
    team_match["wide_action_pct"] = np.where(team_match["passes_total"] > 0,
                                             team_match["wide_actions"] / team_match["passes_total"] * 100, np.nan)
    team_match["high_press_pct"] = np.where(team_match["defensive_duels"] > 0,
                                            team_match["high_press_actions"] / team_match["defensive_duels"] * 100,
                                            np.nan)

    team_match.to_csv("team_match_tactics.csv", index=False)
    return team_match

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

def tactic_cluster(norm,team_id,top_n=5,csv=False):
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans

    norm = norm

    # feature: sadece z-score kolonlari (oyuncu tarafiyla ayni mantik)
    feature_cols = [c for c in norm.columns if c.endswith("_z")]
    X = norm[feature_cols].fillna(0)

    # oyuncudan farkli olarak burada pozisyon grubu yok - butun takimlar
    # tek grup olarak kumeleniyor, lig kucuk oldugu icin (20 takim) az sayida
    # kume yeterli
    n_clusters = 3
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    norm["style_cluster"] = km.fit_predict(X)

    def similar_team(team_id, top_n):
        idx = norm[norm["teamId"] == team_id].index[0]
        sims = cosine_similarity(X.loc[[idx]], X)[0]
        result = norm[["teamId", "style_cluster"]].copy()
        result["similarity"] = sims
        result = result[result["teamId"] != team_id]
        result = result.sort_values("similarity", ascending=False).head(top_n)
        return result

    sonuc = similar_team(team_id, top_n)

    if csv:
        sonuc.to_csv("similar_team.csv", index=False)

    return sonuc

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
