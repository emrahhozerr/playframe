"""
full_pipeline.py
--------
Data Engineer + Data Preprocessing + Feature Engineer asamalarinin
TUMUNU tek dosyada toplayan pipeline. Git'e gonderilecegi icin absolute
Windows path yok - butun yollar bu dosyanin kendi konumuna gore (relative)
hesaplaniyor.

KULLANIM:
1) Bu dosyayla ayni klasore "raw_data" adinda bir klasor ac, ham Wyscout
   dosyalarini (players.csv, teams.csv, tags2name.csv, player_games.csv,
   matches_England.csv, events_England.csv, eventid2name.csv) oraya koy.
2) `python full_pipeline.py` calistir.
3) Butun ara/nihai ciktilar bu dosyayla ayni klasordeki "data" klasorune yazilir.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
DATA_DIR = os.path.join(BASE_DIR, "data")

PLAYERS_CSV = os.path.join(RAW_DATA_DIR, "players.csv")
TEAMS_CSV = os.path.join(RAW_DATA_DIR, "teams.csv")
TAG2NAMES_CSV = os.path.join(RAW_DATA_DIR, "tags2name.csv")
PLAYERS_GAMES_CSV = os.path.join(RAW_DATA_DIR, "player_games.csv")
MATCHES_ENG_CSV = os.path.join(RAW_DATA_DIR, "matches_England.csv")
EVENTS_ENG_CSV = os.path.join(RAW_DATA_DIR, "events_England.csv")
EVENTID2NAME_CSV = os.path.join(RAW_DATA_DIR, "eventid2name.csv")

# matches_eng_pipline icindeki dateutc sutununa eklenecek saat farki
EUTC_OFFSET = 2


# ==========================================================================
# AŞAMA 1: DATA ENGINEER — ham veriyi al, temizle, birlestir
# ==========================================================================

def players_pipline(players):
    """
    BİLGİ ="foot","currentTeamId","currentNationalTeamId" bu üç değişkenin
    eksik değişrleri "unknow" ile doldurdu
    """
    import pandas as pd
    import numpy as np
    import ast
    dataframe = pd.read_csv(players)

    dataframe["role"] = dataframe["role"].apply(ast.literal_eval)
    dataframe["passportArea"] = dataframe["passportArea"].apply(ast.literal_eval)
    dataframe["birthArea"] = dataframe["birthArea"].apply(ast.literal_eval)

    dataframe["role_code"] = dataframe["role"].apply(lambda x: x["code2"])
    dataframe["role_name"] = dataframe["role"].apply(lambda x: x["name"])

    dataframe["passport_country"] = dataframe["passportArea"].apply(lambda x: x["name"])
    dataframe["passport_country_id"] = dataframe["passportArea"].apply(lambda x: x['id'])

    dataframe["birth_country"] = dataframe["birthArea"].apply(lambda x: x["name"])
    dataframe["birth_country_id"] = dataframe["birthArea"].apply(lambda x: x["id"])

    dataframe.drop(["role", "passportArea", "birthArea"], axis=1, inplace=True)

    dataframe["weight"] = dataframe["weight"].replace(0, np.nan)
    dataframe["height"] = dataframe["height"].replace(0, np.nan)

    dataframe["currentNationalTeamId"] = dataframe["currentNationalTeamId"].fillna("unknow")
    dataframe["currentTeamId"] = dataframe["currentTeamId"].fillna("unknow")

    dataframe['height'] = dataframe.groupby('role_code')['height'].transform(
        lambda x: x.fillna(x.median()))
    dataframe['weight'] = dataframe.groupby('role_code')['weight'].transform(
        lambda x: x.fillna(x.median()))

    dataframe["foot"] = dataframe["foot"].fillna("unknow")

    dataframe["birthDate"] = pd.to_datetime(dataframe["birthDate"])

    dataframe["name"] = dataframe["firstName"] + " " + dataframe["lastName"]
    dataframe.drop(["firstName", "lastName"], axis=1, inplace=True)

    dataframe["name"] = dataframe["name"].apply(lambda x: x.encode().decode("unicode_escape"))
    dataframe["shortName"] = dataframe["shortName"].apply(lambda x: x.encode().decode("unicode_escape"))

    dataframe.rename(columns={"name": "player_name", }, inplace=True)

    ordered_columns = [
        "wyId", "player_name", "shortName",
        "role_code", "role_name", "currentTeamId",
        "birthDate", "height", "weight", "foot",
        "birth_country", "passport_country",
    ]

    df_new = dataframe[ordered_columns]
    df_new.to_csv("playerss.csv", index=False)
    return df_new


def teams_pipline(teams):
    import pandas as pd
    import numpy as np
    import ast
    dataframe = pd.read_csv(teams)
    dataframe.rename(columns={"name": "team_name"}, inplace=True)

    dataframe["team_name"] = dataframe["team_name"].apply(lambda x: x.encode().decode("unicode_escape"))
    dataframe["officialName"] = dataframe["officialName"].apply(lambda x: x.encode().decode("unicode_escape"))
    dataframe["city"] = dataframe["city"].apply(lambda x: x.encode().decode("unicode_escape"))

    dataframe["area"] = dataframe["area"].apply(ast.literal_eval)

    dataframe["area_team_conutry"] = dataframe["area"].apply(lambda x: x['name'])
    dataframe["area_team_conutry_id"] = dataframe["area"].apply(lambda x: x['id'])
    dataframe.drop("area", axis=1, inplace=True)

    ordered_team_columns = ['wyId', 'team_name', 'officialName', 'type', 'city', 'area_team_conutry', 'area_team_conutry_id']
    df_new = dataframe[ordered_team_columns]
    df_new.to_csv("teamss.csv", index=False)
    return df_new


def tag2names_pipline(tag2names):
    import pandas as pd
    dataframe = pd.read_csv(tag2names)
    dataframe.to_csv("tags_2_namee.csv", index=False)
    return dataframe


def players_games_pipline(players_games):
    import pandas as pd
    import numpy as np
    dataframe = pd.read_csv(players_games)
    del_cols = ['Unnamed: 0', 'firstname', 'lastname', "jersey_number"]
    dataframe.drop(del_cols, axis=1, inplace=True)

    dataframe.rename(columns={"nickname": "shortName"}, inplace=True)

    dataframe["birth_date"] = pd.to_datetime(dataframe["birth_date"])

    dataframe.loc[dataframe["minutes_played"] <= 0, "minutes_played"] = np.nan
    dataframe.dropna(subset=["minutes_played"], axis=0, inplace=True)

    ordered_cols = [
        'game_id', "team_id", "player_id",
        'player_name', 'shortName', 'birth_date',
        'is_starter', 'minutes_played']

    df_new = dataframe[ordered_cols]
    df_new.to_csv("players_gamee.csv", index=False)
    return df_new


def matches_eng_pipline(matches_eng, eutc):
    """ eut değeri ilgili data sette ne is o girilcek"""
    import pandas as pd
    import numpy as np
    import ast
    dataframe = pd.read_csv(matches_eng)

    drop_cols = dataframe.nunique()[dataframe.nunique() < 2].index
    drop_cols = [col for col in drop_cols if col not in ["duration", "competitionId"]]
    dataframe.drop(drop_cols, axis=1, inplace=True)

    def parse_formation(x):
        if isinstance(x, (list, dict)):
            return x
        if pd.isnull(x):
            return []
        x = x.replace("null", "None")
        return ast.literal_eval(x)

    dataframe["teamsData"] = dataframe["teamsData"].apply(parse_formation)
    formation_cols = dataframe.columns[dataframe.columns.str.contains("formation", regex=False)]

    for col in formation_cols:
        dataframe[col] = dataframe[col].apply(parse_formation)

    dataframe.drop(["team1.formation", "team2.formation"], axis=1, inplace=True)
    dataframe.drop("teamsData", axis=1, inplace=True)

    dataframe["team1_lineup_count"] = dataframe["team1.formation.lineup"].apply(len)
    dataframe["team1_bench_count"] = dataframe["team1.formation.bench"].apply(len)
    dataframe["team2_lineup_count"] = dataframe["team2.formation.lineup"].apply(len)
    dataframe["team2_bench_count"] = dataframe["team2.formation.bench"].apply(len)

    dataframe["team1_sub_count"] = dataframe["team1.formation.substitutions"].apply(len)
    dataframe["team2_sub_count"] = dataframe["team2.formation.substitutions"].apply(len)

    def count_events(player_list, key):
        return sum(1 for p in player_list if p[key] != '0')

    dataframe["team1_yellow_cards"] = dataframe["team1.formation.lineup"].apply(lambda x: count_events(x, "yellowCards")) \
                               + dataframe["team1.formation.bench"].apply(lambda x: count_events(x, "yellowCards"))
    dataframe["team1_red_cards"] = dataframe["team1.formation.lineup"].apply(lambda x: count_events(x, "redCards")) \
                            + dataframe["team1.formation.bench"].apply(lambda x: count_events(x, "redCards"))

    dataframe["team2_yellow_cards"] = dataframe["team2.formation.lineup"].apply(lambda x: count_events(x, "yellowCards")) \
                               + dataframe["team2.formation.bench"].apply(lambda x: count_events(x, "yellowCards"))
    dataframe["team2_red_cards"] = dataframe["team2.formation.lineup"].apply(lambda x: count_events(x, "redCards")) \
                            + dataframe["team2.formation.bench"].apply(lambda x: count_events(x, "redCards"))

    dataframe["team1_own_goals"] = dataframe["team1.formation.lineup"].apply(lambda x: count_events(x, "ownGoals")) \
                            + dataframe["team1.formation.bench"].apply(lambda x: count_events(x, "ownGoals"))
    dataframe["team2_own_goals"] = dataframe["team2.formation.lineup"].apply(lambda x: count_events(x, "ownGoals")) \
                            + dataframe["team2.formation.bench"].apply(lambda x: count_events(x, "ownGoals"))

    dataframe["team1_first_sub_minute"] = dataframe["team1.formation.substitutions"].apply(
        lambda x: min(s["minute"] for s in x) if len(x) > 0 else np.nan)
    dataframe["team1_last_sub_minute"] = dataframe["team1.formation.substitutions"].apply(
        lambda x: max(s["minute"] for s in x) if len(x) > 0 else np.nan)

    dataframe["team2_first_sub_minute"] = dataframe["team2.formation.substitutions"].apply(
        lambda x: min(s["minute"] for s in x) if len(x) > 0 else np.nan)
    dataframe["team2_last_sub_minute"] = dataframe["team2.formation.substitutions"].apply(
        lambda x: max(s["minute"] for s in x) if len(x) > 0 else np.nan)

    drop_cols = [
        'team1.formation.bench', 'team1.formation.lineup', 'team1.formation.substitutions',
        'team2.formation.bench', 'team2.formation.lineup', "team1.coachId", "team2_lineup_count",
        'team2.formation.substitutions', "team2.coachId", 'referees', "team1_lineup_count", ]

    dataframe.drop(columns=drop_cols, axis=1, inplace=True, errors="ignore")

    dataframe['match_name'] = dataframe['label'].str.split(',').str[0].str.replace(' - ', ' vs ')
    dataframe.drop(columns=['label'], inplace=True)

    sub_minute = dataframe.columns[dataframe.columns.str.contains("_sub_minute")]
    dataframe[sub_minute] = dataframe[sub_minute].astype("Int64")

    dataframe["dateutc"] = pd.to_datetime(dataframe["dateutc"])
    dataframe["dateutc"] = dataframe["dateutc"] + pd.Timedelta(hours=eutc)

    ordered_cols = [
        'wyId', 'competitionId', 'gameweek', 'duration', 'dateutc', 'match_name', 'venue', 'winner',
        'team1.teamId', 'team1.side', 'team1.score', 'team1.scoreHT',
        'team2.teamId', 'team2.side', 'team2.score', 'team2.scoreHT',
        'team1_bench_count', 'team2_bench_count', 'team1_yellow_cards', 'team1_red_cards',
        'team2_yellow_cards', 'team2_red_cards', 'team1_own_goals', 'team2_own_goals',
        'team1_sub_count', 'team2_sub_count',
        'team1_first_sub_minute', 'team1_last_sub_minute',
        'team2_first_sub_minute', 'team2_last_sub_minute']

    df_new = dataframe[ordered_cols]
    df_new = df_new.drop(["team1_own_goals", "team2_own_goals"], axis=1)
    df_new.to_csv("matches_engg.csv", index=False)
    return df_new


def events_eng_pipline(events_eng):
    import pandas as pd
    import numpy as np
    import ast
    dataframe = pd.read_csv(events_eng)
    dataframe = dataframe.drop(columns=["pos_orig_x", "pos_orig_y", "pos_dest_x", "pos_dest_y"])

    pattern = r"\{'y':\s*(-?\d+),\s*'x':\s*(-?\d+)\}"
    matches = dataframe["positions"].str.findall(pattern)
    dataframe["pos_orig_y"] = matches.apply(lambda m: int(m[0][0]) if len(m) > 0 else np.nan)
    dataframe["pos_orig_x"] = matches.apply(lambda m: int(m[0][1]) if len(m) > 0 else np.nan)
    dataframe["pos_dest_y"] = matches.apply(lambda m: int(m[1][0]) if len(m) > 1 else np.nan)
    dataframe["pos_dest_x"] = matches.apply(lambda m: int(m[1][1]) if len(m) > 1 else np.nan)

    dataframe.drop(["positions", "tags"], axis=1, inplace=True)

    def parse_tags(x):
        if isinstance(x, list):
            return x
        if pd.isnull(x):
            return []
        return ast.literal_eval(x)

    dataframe["tagsList"] = dataframe["tagsList"].apply(parse_tags)

    exploded = dataframe["tagsList"].explode()
    tag_dummies = pd.crosstab(exploded.index, exploded)
    tag_dummies.columns = [f"tag_{int(c)}" for c in tag_dummies.columns]
    tag_dummies = (tag_dummies > 0).astype(int)

    dataframe = dataframe.join(tag_dummies)
    tag_cols = tag_dummies.columns
    dataframe[tag_cols] = dataframe[tag_cols].fillna(0).astype(int)

    dataframe = dataframe.drop(columns=["tagsList"])

    dataframe.to_csv("event_englandd.csv", index=False)
    return dataframe


def eventid2_name_pipline(eventid2_name):
    import pandas as pd
    dataframe = pd.read_csv(eventid2_name)
    dataframe.to_csv("event_id_2_namee.csv", index=False)
    return dataframe


def all_dataframes_pipline(mat, event):
    """
    Diğer dataframlermi sabit olduğu için sadece ülke bazlı mat,evet dışardan veriyoruz
    """
    import pandas as pd

    matches = pd.read_csv(mat)
    players = pd.read_csv(r"playerss.csv")
    teams = pd.read_csv(r"teamss.csv")
    players_game = pd.read_csv(r"players_gamee.csv")
    event_id_2_name = pd.read_csv(r"event_id_2_namee.csv")
    event_england = pd.read_csv(event)
    events_raw = event_england.copy()
    tags_2_name = pd.read_csv(r"tags_2_namee.csv")

    players_game = players_game.merge(matches.add_prefix("match_"), left_on="game_id",
                                      right_on="match_wyId", how="inner")
    players_extra = players[["wyId", "role_code", "role_name", "currentTeamId",
                             "height", "weight", "foot", "birth_country", "passport_country"]]
    players_game = players_game.merge(players_extra.add_prefix("player_info_"),
                                      left_on="player_id", right_on="player_info_wyId", how="left", )
    players_game = players_game.merge(teams.add_prefix("team_"), left_on="team_id", right_on="team_wyId", how="left")
    players_game = players_game.drop(columns=["match_wyId", "player_info_wyId", "team_wyId"])
    players_game.to_csv("player_game_merged.csv", index=False)

    event_england = event_england.merge(matches.add_prefix("match_"), left_on="matchId",
                                        right_on="match_wyId", how="left")
    players_extra2 = players[["wyId", "player_name", "shortName", "role_code",
                              "role_name", "currentTeamId", "height", "weight", "foot", "birth_country"]]
    event_england = event_england.merge(players_extra2.add_prefix("player_"),
                                        left_on="playerId", right_on="player_wyId", how="left")
    event_england = event_england.merge(teams.add_prefix("team_"), left_on="teamId", right_on="team_wyId", how="left")
    event_england = event_england.merge(event_id_2_name, left_on=["eventId", "subEventId"],
                                        right_on=["event", "subevent"], how="left")
    event_only_labels = event_id_2_name.drop_duplicates("event").set_index("event")["event_label"]
    missing_mask = event_england["event_label"].isna()
    event_england.loc[missing_mask, "event_label"] = event_england.loc[missing_mask, "eventId"].map(event_only_labels)
    event_england = event_england.drop(columns=["match_wyId", "player_wyId", "team_wyId", "event", "subevent"])
    event_england.to_csv("event_england_merged.csv", index=False)

    tag_cols = [col for col in events_raw.columns if col.startswith("tag_")]
    long_tags = events_raw.melt(id_vars=["id"], value_vars=tag_cols,
                                var_name="tag_col", value_name="present")
    long_tags = long_tags[long_tags["present"] == 1]
    long_tags["Tag"] = long_tags["tag_col"].str.replace("tag_", "").astype(int)
    long_tags = long_tags.merge(tags_2_name, on="Tag", how="left")
    long_tags = long_tags.drop(columns=["tag_col", "present"])
    long_tags.to_csv("event_tags_long.csv", index=False)
    return players_game, event_england, long_tags


# ==========================================================================
# AŞAMA 2: DATA PREPROCESSING — birlesmis veri setlerindeki hatalari duzelt
# ==========================================================================

def event_eng_data_preprocessing(dataframe, csv=False):
    import pandas as pd
    import numpy as np

    eng_players_df = dataframe[dataframe['playerId'] != 0].copy()

    eng_players_df["match_gameweek"] = eng_players_df["match_gameweek"].astype("int64")

    eng_players_df.rename(columns={"match_winner": "match_winner_id"}, inplace=True)

    unique_value_tow = [col for col in eng_players_df.columns if eng_players_df[col].nunique() < 2]
    eng_players_df.drop(unique_value_tow, axis=1, inplace=True)

    eng_players_df["match_dateutc"] = pd.to_datetime(eng_players_df["match_dateutc"])

    def compute_winner(row):
        s1, s2 = row["match_team1.score"], row["match_team2.score"]
        if s1 == s2:
            return 0
        elif s1 > s2:
            return row["match_team1.teamId"]
        else:
            return row["match_team2.teamId"]

    eng_players_df["match_winner_id_fixed"] = eng_players_df.apply(compute_winner, axis=1)

    eng_players_df["player_currentTeamId"] = eng_players_df["player_currentTeamId"].replace("unknow", np.nan)
    eng_players_df["player_foot"] = eng_players_df["player_foot"].replace("unknow", np.nan)

    if csv:
        eng_players_df.to_csv("eng_players_data_prep.csv", index=False)
    return eng_players_df


def player_game_data_preprocessing(dataframe, csv=False):
    import pandas as pd
    import numpy as np

    data = dataframe.copy()

    data["match_gameweek"] = data["match_gameweek"].astype("int64")

    data_columsn = ["match_dateutc", "birth_date"]
    for col in data_columsn:
        data[col] = pd.to_datetime(data[col])

    unique_value_toww = [col for col in data.columns if data[col].nunique() < 2]
    data.drop(unique_value_toww, axis=1, inplace=True)

    data.rename(columns={"match_winner": "match_winner_id"}, inplace=True)

    def compute_winner(row):
        s1, s2 = row["match_team1.score"], row["match_team2.score"]
        if s1 == s2:
            return 0
        elif s1 > s2:
            return row["match_team1.teamId"]
        else:
            return row["match_team2.teamId"]

    data["match_winner_id_fixed"] = data.apply(compute_winner, axis=1)

    data["player_info_currentTeamId"] = data["player_info_currentTeamId"].replace("unknow", np.nan)
    data["player_info_foot"] = data["player_info_foot"].replace("unknow", np.nan)

    if csv:
        data.to_csv("player_game_data_prep.csv", index=False)

    return data


# ==========================================================================
# AŞAMA 3: FEATURE ENGINEER — feature uretimi + scout/taktik/gelisim motorlari
# ==========================================================================

def xt_bulider_func(event_eng, csv=False):
    import pandas as pd
    import numpy as np

    usecols = ["id", "playerId", "matchId", "teamId", "eventName", "subEventName", "pos_orig_x",
               "pos_orig_y", "pos_dest_x", "pos_dest_y", "tag_101", "tag_703", "tag_1801"]

    ev = pd.read_csv(event_eng, usecols=usecols)

    n_x = 16
    n_y = 12
    n_zones = n_x * n_y

    ev["orig_zone_x"] = np.minimum((ev["pos_orig_x"] // (100 / n_x)).astype(int), n_x - 1)
    ev["orig_zone_y"] = np.minimum((ev["pos_orig_y"] // (100 / n_y)).astype(int), n_y - 1)
    ev["dest_zone_x"] = np.minimum((ev["pos_dest_x"].fillna(-1) // (100 / n_x)).astype(int), n_x - 1)
    ev["dest_zone_y"] = np.minimum((ev["pos_dest_y"].fillna(-1) // (100 / n_y)).astype(int), n_y - 1)
    ev["orig_zone"] = ev["orig_zone_y"] * n_x + ev["orig_zone_x"]
    ev["dest_zone"] = ev["dest_zone_y"] * n_x + ev["dest_zone_x"]

    is_pass_ok = (ev["eventName"] == "Pass") & (ev["tag_1801"] == 1)
    is_take_on_ok = (ev["subEventName"] == "Ground attacking duel") & (ev["tag_703"] == 1)
    is_move = is_pass_ok | is_take_on_ok
    is_shot = ev["eventName"] == "Shot"
    is_goal = is_shot & (ev["tag_101"] == 1)

    move_counts = ev.loc[is_move, "orig_zone"].value_counts().reindex(range(n_zones), fill_value=0)
    shot_counts = ev.loc[is_shot, "orig_zone"].value_counts().reindex(range(n_zones), fill_value=0)
    goal_counts = ev.loc[is_goal, "orig_zone"].value_counts().reindex(range(n_zones), fill_value=0)
    total_actions = move_counts + shot_counts

    shot_prob = np.where(total_actions > 0, shot_counts / total_actions, 0)
    move_prob = 1 - shot_prob
    goal_prob = np.where(shot_counts > 0, goal_counts / shot_counts, 0)

    transition_counts = (ev.loc[is_move].groupby(["orig_zone", "dest_zone"]).size()
                         .unstack(fill_value=0).reindex(index=range(n_zones), columns=range(n_zones), fill_value=0))

    row_sums = transition_counts.sum(axis=1).replace(0, np.nan)
    transition_matrix = transition_counts.div(row_sums, axis=0).fillna(0).values

    # xT degerlerini iteratif hesapla (orijinal xt_builder.py'deki mantik,
    # global degil fonksiyonun kendi icine tasindi - self-contained olsun diye)
    xt = np.zeros(n_zones)
    n_iterations = 10
    for _ in range(n_iterations):
        xt = shot_prob * goal_prob + move_prob * (transition_matrix @ xt)

    ev["xt_added"] = np.nan
    ev.loc[is_move, "xt_added"] = xt[ev.loc[is_move, "dest_zone"].values] - xt[ev.loc[is_move, "orig_zone"].values]
    ev.loc[is_move, "xt_added"].mean()

    xt_match = (ev.loc[is_move].groupby(["playerId", "matchId"], as_index=False)["xt_added"].sum()
                .rename(columns={"xt_added": "xt_added_total"}))
    if csv:
        xt_match.to_csv("event_xt.csv", index=False)

    return xt_match


def perc90_builder(event_eng, player_game, csv=False):
    import pandas as pd
    import numpy as np

    on_target_tags = [f"tag_{col}" for col in range(1201, 1210)]
    off_target_tags = [f"tag_{col}" for col in range(1210, 1217)]
    post_tags = [f"tag_{col}" for col in range(1217, 1224)]
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

    base_cols = ["playerId", "matchId", "teamId", "eventName", "subEventName"]
    usecols = base_cols + all_tags

    dtype_map = {c: "int8" for c in all_tags}
    dtype_map.update({
        "playerId": "int32", "matchId": "int32", "teamId": "int32",
        "eventName": "category", "subEventName": "category",
    })

    ev = event_eng[usecols].astype(dtype_map)

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

    feat = pd.DataFrame({
        "playerId": ev["playerId"],
        "matchId": ev["matchId"],
        "teamId": ev["teamId"],

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

        "left_foot_actions": ev["tag_401"],
        "right_foot_actions": ev["tag_402"],
        "head_body_actions": ev["tag_403"],

        "free_space_right_actions": ev["tag_501"],
        "free_space_left_actions": ev["tag_502"],

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

        "take_ons": is_take_on.astype("int8"),
        "take_ons_won": (is_take_on & (ev["tag_703"] == 1)).astype("int8"),
        "feints": ev["tag_1301"],

        "interceptions": ev["tag_1401"],
        "sliding_tackles": ev["tag_1601"],
        "clearances": is_clearance.astype("int8"),
        "blocks": ev["tag_2101"],
        "dangerous_ball_lost": ev["tag_2001"],
        "anticipated_against": ev["tag_601"],
        "anticipation_actions": ev["tag_602"],

        "missed_ball": ev["tag_1302"],
        "own_goals": ev["tag_102"],

        "fouls_committed": is_foul.astype("int8"),
        "yellow_cards": ev["tag_1702"],
        "second_yellow_cards": ev["tag_1703"],
        "red_cards_direct": ev["tag_1701"],
        "red_cards": (ev["tag_1701"] + ev["tag_1703"]).clip(upper=1).astype("int8"),
        "fairplay_actions": ev["tag_1001"],

        "shots_total": is_shot.astype("int8"),
        "goals": (is_shot & (ev["tag_101"] == 1)).astype("int8"),
        "goals_conceded_event": ((ev["eventName"] == "Save attempt") & (ev["tag_101"] == 1)).astype("int8"),
        "shots_on_target": (is_shot & is_on_target).astype("int8"),
        "shots_off_target": (is_shot & is_off_target).astype("int8"),
        "shots_post": (is_shot & is_post).astype("int8"),
        "shots_blocked": (is_shot & (ev["tag_2101"] == 1)).astype("int8"),

        "touches": is_touch.astype("int8"),
        "accelerations": is_acceleration.astype("int8"),
        "counter_attack_actions": ev["tag_1901"],
        "offsides": is_offside.astype("int8"),
        "accurate_actions_total": ev["tag_1801"],
        "inaccurate_actions_total": ev["tag_1802"],

        "save_attempts": is_save_attempt.astype("int8"),
        "reflex_saves": is_reflexes.astype("int8"),
        "gk_leaving_line": is_gk_leaving_line.astype("int8"),

        "corners_taken": is_corner.astype("int8"),
        "throw_ins": is_throw_in.astype("int8"),
        "goal_kicks": is_goal_kick.astype("int8"),
    })

    zone_feat = {}
    for tag_col, label in zone_labels.items():
        zone_feat[label] = (is_shot & (ev[tag_col] == 1)).astype("int8")
    feat = pd.concat([feat, pd.DataFrame(zone_feat)], axis=1)

    del ev

    count_cols = [col for col in feat.columns if col not in ("playerId", "matchId", "teamId")]
    agg = feat.groupby(["playerId", "matchId", "teamId"], as_index=False)[count_cols].sum()

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

    pg["match_dateutc"] = pd.to_datetime(pg["match_dateutc"])
    pg["birth_date"] = pd.to_datetime(pg["birth_date"])
    pg["player_age_at_match"] = ((pg["match_dateutc"] - pg["birth_date"]).dt.days / 365.25).round(1)
    pg = pg.drop(columns=["birth_date"])

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

    merged = pg.merge(agg, on=["playerId", "matchId", "teamId"], how="left")
    merged[count_cols] = merged[count_cols].fillna(0)

    minutes = merged["minutes_played"].replace(0, np.nan)
    per90 = merged[count_cols].div(minutes, axis=0) * 90
    per90.columns = [f"{col}_p90" for col in count_cols]

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

    result = pd.concat([merged, per90], axis=1)

    if csv:
        result.to_csv("perc90_builder.csv", index=False)
    return result


def season_normalize_func(per90_1, xt_match, csv=False):
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    per90 = per90_1

    xt_match = xt_match
    per90 = per90.merge(xt_match, on=["playerId", "matchId"], how="left")
    per90["xt_added_total"] = per90["xt_added_total"].fillna(0)

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

    player_meta = df.groupby("playerId", as_index=False).agg(
        player_name=("player_name", "first"),
        role_code=("player_info_role_code", "first"),
        role_name=("player_info_role_name", "first"),
        height=("player_info_height", "first"),
        weight=("player_info_weight", "first"),
        foot=("player_info_foot", "first"),
        birth_country=("player_info_birth_country", "first"),
        current_team_id=("player_info_currentTeamId", "last"))

    season_sum = df.groupby("playerId", as_index=False)[raw_count_cols + ["minutes_played"]].sum()

    matches_played = (
        df.groupby("playerId", as_index=False)["matchId"].nunique().rename(columns={
            "matchId": "matches_played"}))

    season = player_meta.merge(season_sum, on="playerId").merge(matches_played, on="playerId")

    minutes = season["minutes_played"].replace(0, np.nan)
    for col in raw_count_cols:
        season[f"{col}_p90"] = season[col] / minutes * 90

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
    season["right_foot_dominance_pct"] = np.where(foot_total > 0, season["right_foot_actions"] / foot_total * 100, np.nan)

    group_cols = ["role_code"]
    norm = season.copy()
    norm_p90_cols = [col for col in norm.columns if col.endswith("_p90")]
    norm["is_eligible"] = norm["minutes_played"] >= 600
    eligible = norm[norm["is_eligible"]]

    group_stats = eligible.groupby(group_cols)[norm_p90_cols].agg(["mean", "std"])
    group_stats.columns = [f"{col}_{stat}" for col, stat in group_stats.columns]
    norm = norm.merge(group_stats, on=group_cols, how="left")

    for col in norm_p90_cols:
        norm[f"{col}_z"] = (norm[col] - norm[f"{col}_mean"]) / norm[f"{col}_std"]

    norm = norm.drop(columns=[col for col in norm.columns if col.endswith("_mean") or col.endswith("_std")])

    mask = norm["is_eligible"]
    for col in norm_p90_cols:
        norm[f"{col}_percentile"] = np.nan
        norm.loc[mask, f"{col}_percentile"] = norm.loc[mask].groupby(group_cols)[col].rank(pct=True) * 100

    if csv:
        season.to_csv("player_season.csv", index=False)
        norm.to_csv("player_season_normalized.csv", index=False)

    return season, norm


def tactic_builder_func(ev):
    import pandas as pd
    import numpy as np

    ev = pd.read_csv(ev)
    length_m = 105
    width_m = 68

    dx_m = (ev["pos_dest_x"] - ev["pos_orig_x"]) / 100 * length_m
    dy_m = (ev["pos_dest_y"] - ev["pos_orig_y"]) / 100 * width_m
    ev["pass_distance_m"] = np.sqrt(dx_m ** 2 + dy_m ** 2)
    ev["forward_progress"] = ev["pos_dest_x"] - ev["pos_orig_x"]

    is_pass = ev["eventName"] == "Pass"
    is_pass_ok = is_pass & ev["tag_1801"]
    is_long_ball = is_pass & (ev["pass_distance_m"] >= 30)

    is_final_third_entry = is_pass & (ev["pos_orig_x"] < 66) & (ev["pos_dest_x"] >= 66)
    is_box_entry = is_pass & (ev["pos_dest_x"] >= 84) & (ev["pos_dest_y"].between(20, 80))
    is_wide_action = is_pass & ((ev["pos_orig_y"] < 33) | (ev["pos_orig_y"] > 66))

    is_defensive_duel = ev["subEventName"] == "Ground defending duel"
    is_foul = ev["eventName"] == "Foul"
    is_high_press_action = (is_defensive_duel | is_foul) & (ev["pos_orig_x"] >= 60)

    is_set_piece = ev["eventName"] == "Free Kick"
    is_shot = ev["eventName"] == "Shot"
    is_goal = is_shot & ev["tag_101"]

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

    avg_cols = ["avg_pass_distance_m", "avg_forward_progress"]
    season_avg = team_match.groupby("teamId", as_index=False)[avg_cols].mean()

    season = season_sum.merge(matches_played, on="teamId").merge(season_avg, on="teamId")

    season["pass_accuracy_pct"] = np.where(season["passes_total"] > 0,
        season["passes_accurate"] / season["passes_total"] * 100, np.nan)
    season["long_ball_pct"] = np.where(season["passes_total"] > 0,
        season["long_balls"] / season["passes_total"] * 100, np.nan)
    season["wide_action_pct"] = np.where(season["passes_total"] > 0,
        season["wide_actions"] / season["passes_total"] * 100, np.nan)
    season["high_press_pct"] = np.where(season["defensive_duels"] > 0,
        season["high_press_actions"] / season["defensive_duels"] * 100, np.nan)

    for col in raw_count_cols:
        season[f"{col}_per_match"] = season[col] / season["matches_played"]

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


def scout_engine_func(norm, user, head=10):
    """
    SCOUT MOTORU (oyuncu scouting).
    Ne yapar: player_season_normalized icindeki oyunculari (sadece
    minutes_played>=600 sartini saglayan "eligible" havuz) ayni role_code
    (pozisyon) grubu icinde KMeans ile stil kumelerine ayirir, sonra
    "user" olarak verilen oyuncuya cosine similarity ile en cok benzeyen
    "head" kadar oyuncuyu dondurur (ayni pozisyondan, benzer p90/z-score
    profiline sahip alternatif/benzer oyunculari bulmak icin).
    Yan etki: butun havuzu style_cluster etiketiyle "player_season_scouted.csv"
    olarak da kaydeder.
    Kullanim: scout_engine_func(player_norm, "Eden Hazard", head=10)
    """
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans

    norm = norm

    feature_cols = [col for col in norm.columns if col.endswith("_p90_z")]
    X_all = norm[feature_cols].fillna(0)

    pool = norm[norm["is_eligible"]].copy()
    pool_X = X_all.loc[pool.index]

    n_clusters = 4
    pool["style_cluster"] = -1

    for role, idx in pool.groupby("role_code").groups.items():
        X_role = pool_X.loc[idx]
        if len(X_role) < n_clusters:
            continue
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pool.loc[idx, "style_cluster"] = km.fit_predict(X_role)

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

    scout_user = similar_players(user, top_n=head).sort_values("similarity", ascending=False)
    norm = norm.merge(pool[["playerId", "style_cluster"]], on="playerId", how="left")
    norm.to_csv("player_season_scouted.csv", index=False)
    return scout_user


def tactic_cluster(norm, team_id, top_n=5, csv=False):
    """
    TAKTIK MOTORU (takim stili).
    Ne yapar: team_season_tactics_normalized icindeki butun takimlari
    (20 takim, tek grup - pozisyon ayrimi yok) taktik z-score profiline
    gore KMeans ile 3 stil kumesine ayirir, sonra "team_id" ile verilen
    takima cosine similarity acisindan en cok benzeyen "top_n" takimi
    dondurur (benzer oyun tarzina sahip rakip/emsal takimlari bulmak icin).
    csv=True verilirse sonucu "similar_team.csv" olarak da kaydeder.
    Kullanim: tactic_cluster(team_norm, team_id=1611, top_n=5, csv=True)
    """
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans

    norm = norm

    feature_cols = [c for c in norm.columns if c.endswith("_z")]
    X = norm[feature_cols].fillna(0)

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


def player_radar_func(norm, player_names, top_n=25):
    """
    OYUNCU GELISIMI / gorsellestirme.
    Ne yapar: verilen oyuncu isim(ler)i icin pozisyona ozel (FW/MD/DF/GK
    icin farkli metrik setleri) percentile degerleriyle bir radar grafik
    cizer ve "player_radar_<isim(ler)>.png" olarak kaydeder (birden fazla
    isim verilirse ustuste karsilastirmali cizer). Tek bir oyuncu adi
    verildiyse ayrica o oyuncunun butun p90 metriklerinin percentile
    siralamasini gosteren bir bar chart daha cizip
    "player_full_stats_<isim>.png" olarak kaydeder.
    Kullanim: player_radar_func(player_norm, "Eden Hazard")
              player_radar_func(player_norm, ["Eden Hazard", "Mohamed Salah"])
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    norm = norm

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
            "goals_conceded_event_p90": "Az Gol Yeme",
        },
    }

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

    if isinstance(player_names, str):
        player_names = [player_names]

    radar_path = plot_radar(player_names)

    stat_path = None
    if len(player_names) == 1:
        stat_path = full_stat_sheet(player_names[0], top_n)

    return radar_path, stat_path


def pass_network(ev, match_id, team_id):
    """
    PAS AGI / bolge yogunlugu gorsellestirmesi.
    Ne yapar: verilen "match_id" + "team_id" icin o takimin o mactaki
    butun isabetli paslarini bulur, oyuncularin saha uzerindeki ortalama
    pozisyonlarini hesaplar ve iki oyuncu arasindaki pas sayisi kadar
    kalin cizgilerle bir "pas agi" haritasi cizer (solda); sagda ise ayni
    takimin o mactaki tum aksiyonlarinin saha uzerindeki yogunluk
    haritasini (hexbin) cizer. Ikisini tek bir "pass_network.png"
    dosyasina kaydeder.
    Kullanim: pass_network(event_df, match_id=2499719, team_id=1631)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    ev = ev
    match_id = match_id
    team_id = team_id

    team_ev = ev[(ev["matchId"] == match_id) & (ev["teamId"] == team_id)].copy()
    team_ev = team_ev.sort_values(["matchPeriod", "eventSec"]).reset_index(drop=True)
    is_pass_ok = (team_ev["eventName"] == "Pass") & (team_ev["tag_1801"] == 1)

    team_ev["receiver_playerId"] = team_ev["playerId"].shift(-1)

    passes = team_ev[is_pass_ok].copy()
    passes = passes.dropna(subset=["receiver_playerId"])
    passes["receiver_playerId"] = passes["receiver_playerId"].astype(int)

    passes = passes[passes["playerId"] != passes["receiver_playerId"]]

    avg_pos = team_ev.groupby("playerId")[["pos_orig_x", "pos_orig_y"]].mean()

    pair_counts = {}
    for _, row in passes.iterrows():
        a, b = row["playerId"], row["receiver_playerId"]
        key = tuple(sorted([a, b]))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    pass_volume = passes["playerId"].value_counts().add(passes["receiver_playerId"].value_counts(), fill_value=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8))

    for ax in (ax1, ax2):
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 68)
        ax.add_patch(plt.Rectangle((0, 0), 105, 68, fill=False, color="black"))
        ax.axvline(52.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_aspect("equal")

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

    x_m = team_ev["pos_orig_x"] / 100 * 105
    y_m = team_ev["pos_orig_y"] / 100 * 68
    hb = ax2.hexbin(x_m, y_m, gridsize=18, extent=(0, 105, 0, 68), cmap="Reds", mincnt=1)
    fig.colorbar(hb, ax=ax2, label="aksiyon sayisi")
    ax2.set_title(f"Bolge Yogunlugu - matchId={match_id}, teamId={team_id}")

    plt.tight_layout()
    plt.savefig("pass_network.png", dpi=150)
    plt.close(fig)


# ==========================================================================
# MAIN — 3 asamayi sirayla calistirir
# ==========================================================================

def main():
    import pandas as pd

    os.makedirs(DATA_DIR, exist_ok=True)
    os.chdir(DATA_DIR)

    print("========== AŞAMA 1: Data Engineer ==========")
    print("1/8 - players_pipline çalışıyor...")
    players_pipline(PLAYERS_CSV)
    print("2/8 - teams_pipline çalışıyor...")
    teams_pipline(TEAMS_CSV)
    print("3/8 - tag2names_pipline çalışıyor...")
    tag2names_pipline(TAG2NAMES_CSV)
    print("4/8 - players_games_pipline çalışıyor...")
    players_games_pipline(PLAYERS_GAMES_CSV)
    print("5/8 - matches_eng_pipline çalışıyor...")
    matches_eng_pipline(MATCHES_ENG_CSV, EUTC_OFFSET)
    print("6/8 - events_eng_pipline çalışıyor...")
    events_eng_pipline(EVENTS_ENG_CSV)
    print("7/8 - eventid2_name_pipline çalışıyor...")
    eventid2_name_pipline(EVENTID2NAME_CSV)
    print("8/8 - all_dataframes_pipline çalışıyor...")
    players_game, event_england, long_tags = all_dataframes_pipline(
        mat="matches_engg.csv", event="event_englandd.csv")
    print("AŞAMA 1 tamamlandı.\n")

    print("========== AŞAMA 2: Data Preprocessing ==========")
    print("1/2 - event_eng_data_preprocessing çalışıyor...")
    event_df = event_eng_data_preprocessing(event_england, csv=True)
    print("2/2 - player_game_data_preprocessing çalışıyor...")
    player_game_df = player_game_data_preprocessing(players_game, csv=True)
    print("AŞAMA 2 tamamlandı.\n")

    print("========== AŞAMA 3: Feature Engineer ==========")
    print("1/6 - xt_bulider_func çalışıyor...")
    xt_match = xt_bulider_func("eng_players_data_prep.csv", csv=True)
    print("2/6 - perc90_builder çalışıyor...")
    per90 = perc90_builder(event_df, player_game_df, csv=True)
    print("3/6 - season_normalize_func çalışıyor...")
    season, norm = season_normalize_func(per90, xt_match, csv=True)
    print("4/6 - tactic_builder_func çalışıyor...")
    team_match = tactic_builder_func("eng_players_data_prep.csv")
    print("5/6 - tactic_season_normalize_func çalışıyor...")
    team_season, team_norm = tactic_season_normalize_func(team_match, csv=True)
    print("AŞAMA 3 tamamlandı.\n")

    print("Tüm pipeline (3 aşama) tamamlandı.")
    print(f"norm (player) shape: {norm.shape}")
    print(f"team_norm shape    : {team_norm.shape}")

    return event_df, norm, team_norm


if __name__ == "__main__":
    event_df, player_norm, team_norm = main()

# asagidakiler otomatik akisin parcasi degil - spesifik oyuncu/takim/mac
# sorgusu gerektiriyorlar, main() bittikten sonra ihtiyaca gore elle cagrilir.
# Her satirin ne yaptigi ayrica ilgili fonksiyonun docstring'inde anlatiliyor.
# Hangi CSV'yi okuduklarini da yaninda belirttim - main() calismis, "data/"
# klasorunde bu dosyalar hazir olmali.

# SCOUT MOTORU  -> okudugu: player_season_normalized.csv
# "Eden Hazard"a ayni pozisyondan en cok benzeyen 10 oyuncuyu bulur
# scout_engine_func(player_norm, "Eden Hazard", head=10)

# OYUNCU GELISIMI  -> okudugu: player_season_normalized.csv
# "Eden Hazard" icin radar grafik + tam istatistik bar chart cizer
# player_radar_func(player_norm, "Eden Hazard")

# TAKTIK MOTORU  -> okudugu: team_season_tactics_normalized.csv
# team_id=1611 olan takima taktik/stil olarak en cok benzeyen 5 takimi bulur
# tactic_cluster(team_norm, team_id=1611, top_n=5, csv=True)

# PAS AGI  -> okudugu: eng_players_data_prep.csv
# match_id=2499719 macinda team_id=1631 takiminin pas agi + bolge yogunluk haritasini cizer
# pass_network(event_df, match_id=2499719, team_id=1631)
