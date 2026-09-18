import pandas as pd

pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("expand_frame_repr", False)
pd.set_option("float_format", lambda x: "%.3f" % x)

events_df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data Enginers\event_england_merged.csv", low_memory=False)
games_df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data Enginers\player_game_merged.csv")
tags_df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data Enginers\event_tags_long.csv")

events_df.shape
games_df.shape

# matchId / game_id birbiryle uyuşuyormu
events_matches = set(events_df["matchId"].unique())
games_matches = set(games_df["game_id"].unique())

len(events_matches)
len(games_matches)
len(events_matches - games_matches)
len(games_matches - events_matches)

# teamId / team_id birbiriyle uyuşuyormu
events_teams = set(events_df["teamId"].unique())
games_teams = set(games_df["team_id"].unique())

len(events_teams)
len(games_teams)
len(events_teams - games_teams)
len(games_teams - events_teams)

# playerId / player_id birbiriyle uyuşuyormu
# playerId=0 - events_df içinde oyuncusuz sistemsel event sentinel değeri gerçek oyuncu değil
# bir oyuncu vardı eda 1 dk oyunua girmiş ikisi arası 1 fark o oyuncudan kaynaklanıyor silcez unutma
events_players = set(events_df.loc[events_df["playerId"] != 0, "playerId"].unique())
games_players = set(games_df["player_id"].unique())

len(events_players)
len(games_players)
len(events_players - games_players)
len(games_players - events_players)

# Event'te olup games_df'de olmayan maç, oyuncu kombinasyonları
# NOT: events_df'te olup games_df'de olmayan 12 kombinasyonu var.
# Bunların bir kısmı gerçekten önemsiz 1 event üreten çok geç/kısa süreli değişiklikler:
# playerId 8686, 134294, 434159, 71654, 91381, 8049  sadece 1 er event.
# AMA 3 tanesi ÖNEMLİ bir eksik playerId 3326 (15 event), 8958 (27 event, Jay Rodriguez,
# match 2499906), 25393 (23 event, Dejan Lovren, match 2500032) -> bu oyuncular maçın
# büyük bölümünde aktif oynamış ama games_df'de o maça ait satırları YOK.
# Kadronun geneli bozuk değil (her maçta sadece 1 oyuncu eksik, doğrulandı), izole ama
# gerçek satır kaybı. Kaynağı muhtemelen lineup verisinin çekildiği pipeline adımı.
# Feature engineering'e geçmeden önce bu 12 kombinasyonu ya tamamla ya da bilinen eksik
# olarak işaretleyip modelden çıkar.

events_combos = set(
    events_df.loc[events_df["playerId"] != 0, ["matchId", "playerId"]]
    .drop_duplicates()
    .itertuples(index=False, name=None)
)
games_combos = set(
    games_df[["game_id", "player_id"]]
    .drop_duplicates()
    .itertuples(index=False, name=None)
)

missing_combos = events_combos - games_combos
len(events_combos)
len(missing_combos)
list(missing_combos)[:10]

# Ortak kolonların iki dosyada tutarlılığı
# team_team_name, team_officialName, team_city, team_area_team_conutry, team_area_team_conutry_id, team_type -> takım seviyesi, ayrı kontrol edilecek
team_cols = ['team_team_name', 'team_officialName', 'team_city', 'team_area_team_conutry', 'team_area_team_conutry_id', 'team_type']
shared_cols = sorted(set(events_df.columns) & set(games_df.columns))
shared_cols = [c for c in shared_cols if c not in team_cols]
shared_cols

events_match_level = events_df[["matchId"] + shared_cols].drop_duplicates(subset=["matchId"])
games_match_level = games_df[["game_id"] + shared_cols].drop_duplicates(subset=["game_id"])

merged_df = events_match_level.merge(
    games_match_level, left_on="matchId", right_on="game_id", suffixes=("_events", "_games")
)

# Her ortak kolonu tek tek kontrol et
# NOT: NaN != NaN pandas'ta her zaman True döner, iki taraf da boşsa bile "farklı" görünür.
# Bu yüzden ikisi de NaN olan satırları false positive olarak elemek gerekiyor.
for col in shared_cols:
    col_events, col_games = f"{col}_events", f"{col}_games"
    if col_events not in merged_df.columns:
        continue
    both_nan = merged_df[col_events].isna() & merged_df[col_games].isna()
    mismatch = merged_df[(merged_df[col_events] != merged_df[col_games]) & ~both_nan]
    if len(mismatch) > 0:
        print(f"UYUŞMUYOR -> {col}: {len(mismatch)} maçta farklı değer")
    else:
        print(f"OK -> {col}: tüm maçlarda tutarlı")

# belirli bir kolonun uyuşmayan satırlarına bakalım
col = "match_team1.score"
mismatch = merged_df[merged_df[f"{col}_events"] != merged_df[f"{col}_games"]]
print(mismatch[["matchId", f"{col}_events", f"{col}_games"]])

# team_cols için ayrı, teamId bazlı kontrol
events_team_level = events_df[["teamId"] + team_cols].drop_duplicates(subset=["teamId"])
games_team_level = games_df[["team_id"] + team_cols].drop_duplicates(subset=["team_id"])
team_merged = events_team_level.merge(
    games_team_level, left_on="teamId", right_on="team_id", suffixes=("_events", "_games")
)
for col in team_cols:
    col_events, col_games = f"{col}_events", f"{col}_games"
    both_nan = team_merged[col_events].isna() & team_merged[col_games].isna()
    mismatch = team_merged[(team_merged[col_events] != team_merged[col_games]) & ~both_nan]
    if len(mismatch) > 0:
        print(f"UYUŞMUYOR -> {col}: {len(mismatch)} / {len(team_merged)} takımda farklı değer")
    else:
        print(f"OK -> {col}: tüm takımlarda tutarlı ({len(team_merged)} takım)")

# tag_101 goal event sayısı vs skor bilgi amaçlı, own-goal'larda tekli sayım olabilir
match_scores = events_df[["matchId", "match_team1.score", "match_team2.score"]].drop_duplicates().set_index("matchId")
total_score = match_scores["match_team1.score"] + match_scores["match_team2.score"]

goal_tag_events = events_df[events_df["tag_101"] == 1].groupby("matchId").size()

goal_check_df = pd.DataFrame({"goal_tag_events": goal_tag_events, "total_score": total_score})
goal_check_df["expected_2x"] = goal_check_df["total_score"] * 2

goal_check_df["goal_tag_events"] = goal_check_df["goal_tag_events"].fillna(0)

odd_count = (goal_check_df["goal_tag_events"] % 2 != 0).sum()

len(goal_check_df)
odd_count
goal_check_df[goal_check_df["goal_tag_events"] % 2 != 0].head(10)