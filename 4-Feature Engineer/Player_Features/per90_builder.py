import pandas as pd
import numpy as np


pd.set_option("display.max_columns",500)
pd.set_option("display.width",1000)
pd.set_option("expand_frame_repr",False)
pd.set_option("float_format",lambda x: "%.3f" % x)

event_eng =  pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Feature_Engineer\Data Preprocessing\data\eng_players_data_prep.csv")
player_game = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Feature_Engineer\Data Preprocessing\data\player_game_data_prep.csv")

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
len(all_zone_tags)

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

len(all_tags)


# ham event verisini isle
# event_eng zaten yukarida tum kolonlariyla okundu; burada sadece ihtiyacimiz
# olan kolonlari secip dtype'lari optimize ediyoruz
base_cols = ["playerId", "matchId", "teamId", "eventName", "subEventName"]
usecols = base_cols + all_tags

dtype_map = {c: "int8" for c in all_tags}
dtype_map.update({
    "playerId": "int32", "matchId": "int32", "teamId": "int32",
    "eventName": "category", "subEventName": "category",
})

ev = event_eng[usecols].astype(dtype_map)
ev.shape


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

is_pass.sum()
is_duel.sum()
is_shot.sum()


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

feat.shape


# goal sayısı dogru mu
wrong_goals = ev["tag_101"]
wrong_goals.sum()
feat["goals"].sum()

# artik ham tabloya ihtiyac yok, bellekten silelim
del ev


# groupby - oyuncu-mac bazinda toplama  birleşitrme öncesi
count_cols = [col for col in feat.columns if col not in ("playerId", "matchId", "teamId")]
agg = feat.groupby(["playerId", "matchId", "teamId"], as_index=False)[count_cols].sum()
agg.shape

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
pg.shape


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
merged.shape


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
    merged["accurate_actions_total"] / (merged["accurate_actions_total"] + merged["inaccurate_actions_total"]) * 100,
    np.nan,
)
merged["duel_win_pct"] = np.where(merged["duels_total"] > 0, merged["duels_won"] / merged["duels_total"] * 100, np.nan)
merged["duel_loss_pct"] = np.where(merged["duels_total"] > 0, merged["duels_lost"] / merged["duels_total"] * 100, np.nan)
merged["duel_neutral_pct"] = np.where(merged["duels_total"] > 0, merged["duels_neutral"] / merged["duels_total"] * 100, np.nan)
merged["take_on_success_pct"] = np.where(merged["take_ons"] > 0, merged["take_ons_won"] / merged["take_ons"] * 100, np.nan)
merged["shot_accuracy_pct"] = np.where(merged["shots_total"] > 0, merged["shots_on_target"] / merged["shots_total"] * 100, np.nan)
merged["shot_conversion_pct"] = np.where(merged["shots_total"] > 0, merged["goals"] / merged["shots_total"] * 100, np.nan)
foot_total = merged["left_foot_actions"] + merged["right_foot_actions"]
merged["right_foot_dominance_pct"] = np.where(foot_total > 0, merged["right_foot_actions"] / foot_total * 100, np.nan)


# ham + per90 + oran tek tabloda birlestir
result = pd.concat([merged, per90], axis=1)
result.shape



# yeni oluşturdğumu veri setinde tekrar eden kayıt varmı baktık
result.duplicated(subset=["playerId", "matchId"]).sum()
# toplam gol sayısı
result["goals"].sum()
# en çok gol atan  oyuncu
result.groupby("player_name")["goals"].sum().sort_values(ascending=False).head()
# eşleşme kontrolü
(result["teamId"] == result["opponent_team_id"]).sum()


# kaydet
result.to_csv("perc90_builder.csv", index=False)

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
