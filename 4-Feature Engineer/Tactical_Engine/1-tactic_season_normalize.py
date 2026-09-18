import pandas as pd
import numpy as np
import warnings

pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

ev = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Feature_Engineer\Data Preprocessing\data\eng_players_data_prep.csv")
ev.shape


# saha olcekleri pitch 0-100 grid, gercek metreye ceviriyoruz
length_m = 105
width_m = 68

dx_m = (ev["pos_dest_x"] - ev["pos_orig_x"]) / 100 * length_m
dy_m = (ev["pos_dest_y"] - ev["pos_orig_y"]) / 100 * width_m
ev["pass_distance_m"] = np.sqrt(dx_m**2 + dy_m**2)
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

is_pass.sum()
is_pass_ok.sum()
is_final_third_entry.sum()
is_box_entry.sum()
is_high_press_action.sum()

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
    team_match["high_press_actions"] / team_match["defensive_duels"] * 100, np.nan)

team_match.shape
team_match.to_csv("team_match_tactics.csv", index=False)

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
