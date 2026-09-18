import pandas as pd

pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.3f" % x)


matches = pd.read_csv(r"matches_engg.csv")
players = pd.read_csv(r"playerss.csv")
teams = pd.read_csv(r"teamss.csv")
players_game = pd.read_csv(r"players_gamee.csv")
event_id_2_name = pd.read_csv(r"event_id_2_namee.csv")

######################################################################
# TABLO 1: OYUNCU-MAÇ BAZLI (players_game + matches + players + teams)
######################################################################
# Maç bilgisini birleştirme
players_game = players_game.merge(matches.add_prefix("match_"),left_on="game_id",
                                  right_on="match_wyId",how="inner")
players_game.head()
players_game.shape

# Oyuncu bilgisini birleştir (players_game'de zaten olmayan ek alanlar)
players_extra = players[["wyId", "role_code", "role_name", "currentTeamId",
                         "height", "weight", "foot", "birth_country", "passport_country"]]

players_game = players_game.merge(players_extra.add_prefix("player_info_"),
                                  left_on="player_id", right_on="player_info_wyId",how="left",)

# Takım bilgisini birleştir
players_game = players_game.merge(teams.add_prefix("team_"),left_on="team_id", right_on="team_wyId",how="left")

# Gereksiz join anahtarlarını temizle
players_game = players_game.drop(columns=["match_wyId", "player_info_wyId", "team_wyId"])

players_game.to_csv("player_game_merged.csv", index=False)
players_game.head()
players_game_.shape
#######################################################
# TABLO 2: EVENT (event_england + matches + players + teams + event_id_2_name)
####################################################
event_england = pd.read_csv("event_englandd.csv")

# Maç bilgisini birleştir
event_england = event_england.merge(matches.add_prefix("match_"),left_on="matchId", right_on="match_wyId",how="left")

# Oyuncu bilgisini birleştir (playerId == 0 -> oyuncu belirlenmemiş, NaN kalır)
players_extra2 = players[["wyId", "player_name", "shortName", "role_code",
                          "role_name","currentTeamId", "height", "weight", "foot", "birth_country"]]

event_england = event_england.merge(players_extra2.add_prefix("player_"),
                                    left_on="playerId", right_on="player_wyId",how="left")

# Takım bilgisini birleştir
event_england = event_england.merge(teams.add_prefix("team_"),left_on="teamId", right_on="team_wyId",how="left")

# Event/subevent etiketlerini birleştir
# Not: eventId=6 (Offside) satırlarında subEventId boş geliyor ama event_id_2_name'de
# subevent=60 olarak tanımlı -> önce 2 anahtarlı (eventId, subEventId) merge yapılır,
# sonra event_label boş kalan satırlar için sadece eventId üzerinden düzeltme yapılır.

event_england = event_england.merge(event_id_2_name,left_on=["eventId", "subEventId"],
                                    right_on=["event", "subevent"],how="left")

event_only_labels = event_id_2_name.drop_duplicates("event").set_index("event")["event_label"]
missing_mask = event_england["event_label"].isna()
event_england.loc[missing_mask, "event_label"] = event_england.loc[missing_mask, "eventId"].map(event_only_labels)

event_england = event_england.drop(columns=["match_wyId", "player_wyId", "team_wyId", "event", "subevent"])

event_england.to_csv("event_england_merged.csv", index=False)

event_england.head()
event_england.shape

############################################
# TABLO 3: TAG'LER
############################################
events_raw = pd.read_csv(r"event_englandd.csv")
tags_2_name = pd.read_csv(r"tags_2_namee.csv")

tag_cols = [col for col in events_raw.columns if col.startswith("tag_")]

long_tags = events_raw.melt(
    id_vars=["id"],
    value_vars=tag_cols,
    var_name="tag_col",
    value_name="present",
)
long_tags = long_tags[long_tags["present"] == 1]
long_tags["Tag"] = long_tags["tag_col"].str.replace("tag_", "").astype(int)

long_tags = long_tags.merge(tags_2_name, on="Tag", how="left")
long_tags = long_tags.drop(columns=["tag_col", "present"])

long_tags.to_csv("event_tags_long.csv", index=False)
long_tags.head()
long_tags.shape


def all_dataframes_pipline(mat, event):
    """
    Diğer dataframlermi sabit olduğu için sadece ülke bazlı mat,evet dışardan veriyoruz
    """

    import pandas as pd

    # data_setler join edilen #
    matches = pd.read_csv(mat)
    players = pd.read_csv(r"playerss.csv")
    teams = pd.read_csv(r"teamss.csv")
    players_game = pd.read_csv(r"players_gamee.csv")
    event_id_2_name = pd.read_csv(r"event_id_2_namee.csv")
    event_england = pd.read_csv(event)
    events_raw = event_england.copy()
    tags_2_name = pd.read_csv(r"tags_2_namee.csv")
    ######################################################################
    # TABLO 1: OYUNCU-MAÇ BAZLI (players_game + matches + players + teams)
    ######################################################################
    # Maç bilgisini birleştirme
    players_game = players_game.merge(matches.add_prefix("match_"), left_on="game_id",
                                      right_on="match_wyId", how="inner")
    # Oyuncu bilgisini birleştir (players_game'de zaten olmayan ek alanlar)
    players_extra = players[["wyId", "role_code", "role_name", "currentTeamId",
                             "height", "weight", "foot", "birth_country", "passport_country"]]

    players_game = players_game.merge(players_extra.add_prefix("player_info_"),
                                      left_on="player_id", right_on="player_info_wyId", how="left", )
    # Takım bilgisini birleştir
    players_game = players_game.merge(teams.add_prefix("team_"), left_on="team_id", right_on="team_wyId", how="left")
    # Gereksiz join anahtarlarını temizle
    players_game = players_game.drop(columns=["match_wyId", "player_info_wyId", "team_wyId"])
    players_game.to_csv("player_game_merged.csv", index=False)
    #######################################################
    # TABLO 2: EVENT (event_england + matches + players + teams + event_id_2_name)
    ####################################################
    # Maç bilgisini birleştir
    event_england = event_england.merge(matches.add_prefix("match_"), left_on="matchId",
                                        right_on="match_wyId",how="left")
    # Oyuncu bilgisini birleştir (playerId == 0 -> oyuncu belirlenmemiş, NaN kalır)
    players_extra2 = players[["wyId", "player_name", "shortName", "role_code",
                              "role_name", "currentTeamId", "height", "weight", "foot", "birth_country"]]
    event_england = event_england.merge(players_extra2.add_prefix("player_"),
                                        left_on="playerId", right_on="player_wyId", how="left")
    # Takım bilgisini birleştir
    event_england = event_england.merge(teams.add_prefix("team_"), left_on="teamId", right_on="team_wyId", how="left")
    # Event/subevent etiketlerini birleştir
    # Not: eventId=6 (Offside) satırlarında subEventId boş geliyor ama event_id_2_name'de
    # subevent=60 olarak tanımlı  önce 2 anahtarlı (eventId, subEventId) merge yapılır,
    # sonra event_label boş kalan satırlar için sadece eventId üzerinden düzeltme yapılır.
    event_england = event_england.merge(event_id_2_name, left_on=["eventId", "subEventId"],
                                        right_on=["event", "subevent"], how="left")
    event_only_labels = event_id_2_name.drop_duplicates("event").set_index("event")["event_label"]
    missing_mask = event_england["event_label"].isna()
    event_england.loc[missing_mask, "event_label"] = event_england.loc[missing_mask, "eventId"].map(event_only_labels)
    event_england = event_england.drop(columns=["match_wyId", "player_wyId", "team_wyId", "event", "subevent"])
    event_england.to_csv("event_england_merged.csv", index=False)
    ############################################
    # TABLO 3: TAG'LER
    ############################################
    tag_cols = [col for col in events_raw.columns if col.startswith("tag_")]
    long_tags = events_raw.melt(id_vars=["id"],value_vars=tag_cols,
                                var_name="tag_col",value_name="present")
    long_tags = long_tags[long_tags["present"] == 1]
    long_tags["Tag"] = long_tags["tag_col"].str.replace("tag_", "").astype(int)
    long_tags = long_tags.merge(tags_2_name, on="Tag", how="left")
    long_tags = long_tags.drop(columns=["tag_col", "present"])
    long_tags.to_csv("event_tags_long.csv", index=False)
    return players_game, event_england, long_tags


mat = r"matches_engg.csv"
event = r"event_englandd.csv"

all_dataframes_pipline(mat, event)




















