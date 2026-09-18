import pandas as pd
import numpy as np
import ast

from pandas.conftest import index

pd.set_option("display.width",1000)
pd.set_option("display.max_columns",500)
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.float_format",lambda x: "%.3f" % x)

df = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data\first_Data\matches_England.csv")
df.head()
df.shape
df.nunique().sort_values(ascending=False)
df.isnull().sum().sort_values(ascending=False)

####################################################
# 2'den az eşsiz değerli sütunları sil #
####################################################
drop_cols = df.nunique()[df.nunique() < 2].index

# yanılgı olmasın diye bir daha kontrol ettik
for col in drop_cols:
    print(df[col].value_counts())

#duration değişkeni uzatma devresine gidilmişmi bilgisi veriyor,
#competitionId liglerin İd  sini veriyor bu yüzden silmiyoruz.
drop_cols = [col for col in drop_cols if col not in ["duration","competitionId"]]

#geri kalanı siliyoruz
df.drop(drop_cols, axis=1, inplace=True)

###################################################
# Sözlük yapılarını açma ve değişlenlere çevirme
###################################################
# Nan değerl olduğu için sözlük yapısnı çözmede sıkıntı yaşadık bu yüzden fonsiyon yazdık
def parse_formation(x):
    if isinstance(x, (list, dict)):
        return x
    if pd.isnull(x):
        return []
    x = x.replace("null", "None")
    return ast.literal_eval(x)

# teamsData sütununu çevir
df["teamsData"] = df["teamsData"].apply(parse_formation)

# Hem team1 hem team2'nin formation ile ilgili TÜM sütunlarını bul ve çevir
formation_cols = df.columns[df.columns.str.contains("formation", regex=False)]

for col in formation_cols:
    df[col] = df[col].apply(parse_formation)

# Kontrol: hepsi dict/list mi oldu?
for col in formation_cols:
    print(col, "-", type(df[col].iloc[0]))


# Kontrol Et aynı bilgiyimi içeriyormu bu sütunlar
df["team1.formation"].iloc[0]["bench"] == df["team1.formation.bench"].iloc[0]
df["team1.formation"].iloc[0]["lineup"] == df["team1.formation.lineup"].iloc[0]
df["team1.formation"].iloc[0]["substitutions"] == df["team1.formation.substitutions"].iloc[0]

# aynı bilgiyi içerdiği için sildik
df.drop(["team1.formation", "team2.formation"], axis=1, inplace=True)

# gereksiz bilgimi içeriyor "teasmdata" seti
df["teamsData"].iloc[0]
df.drop("teamsData", axis=1, inplace=True)

# Kadro büyüklükleri (kaç kişi ilk 11'de, kaç kişi yedekte)
df["team1_lineup_count"] = df["team1.formation.lineup"].apply(len)
df["team1_bench_count"]  = df["team1.formation.bench"].apply(len)
df["team2_lineup_count"] = df["team2.formation.lineup"].apply(len)
df["team2_bench_count"]  = df["team2.formation.bench"].apply(len)

# Kaç değişiklik yapıldı
df["team1_sub_count"] = df["team1.formation.substitutions"].apply(len)
df["team2_sub_count"] = df["team2.formation.substitutions"].apply(len)

# Sarı / kırmızı kart SAYISI (değer '0' değilse, o oyuncu kart görmüş demektir - SAYIYORUZ, TOPLAMIYORUZ)
def count_events(player_list, key):
    return sum(1 for p in player_list if p[key] != '0')

df["team1_yellow_cards"] = df["team1.formation.lineup"].apply(lambda x: count_events(x, "yellowCards")) \
                          + df["team1.formation.bench"].apply(lambda x: count_events(x, "yellowCards"))
df["team1_red_cards"]    = df["team1.formation.lineup"].apply(lambda x: count_events(x, "redCards")) \
                          + df["team1.formation.bench"].apply(lambda x: count_events(x, "redCards"))

df["team2_yellow_cards"] = df["team2.formation.lineup"].apply(lambda x: count_events(x, "yellowCards")) \
                          + df["team2.formation.bench"].apply(lambda x: count_events(x, "yellowCards"))
df["team2_red_cards"]    = df["team2.formation.lineup"].apply(lambda x: count_events(x, "redCards")) \
                          + df["team2.formation.bench"].apply(lambda x: count_events(x, "redCards"))

# Kendi kalesine gol SAYISI (aynı mantık - '0' değilse gerçekleşmiş demektir)
df["team1_own_goals"] = df["team1.formation.lineup"].apply(lambda x: count_events(x, "ownGoals")) \
                       + df["team1.formation.bench"].apply(lambda x: count_events(x, "ownGoals"))
df["team2_own_goals"] = df["team2.formation.lineup"].apply(lambda x: count_events(x, "ownGoals")) \
                       + df["team2.formation.bench"].apply(lambda x: count_events(x, "ownGoals"))

# İlk / son değişiklik dakikası
df["team1_first_sub_minute"] = df["team1.formation.substitutions"].apply(
    lambda x: min(s["minute"] for s in x) if len(x) > 0 else np.nan)
df["team1_last_sub_minute"] = df["team1.formation.substitutions"].apply(
    lambda x: max(s["minute"] for s in x) if len(x) > 0 else np.nan)

df["team2_first_sub_minute"] = df["team2.formation.substitutions"].apply(
    lambda x: min(s["minute"] for s in x) if len(x) > 0 else np.nan)
df["team2_last_sub_minute"] = df["team2.formation.substitutions"].apply(
    lambda x: max(s["minute"] for s in x) if len(x) > 0 else np.nan)
df.head()

# Kartlar şüpeli geldi onun sağlamlığına bakıyoruz
all_owngoal_values = []
for player_list in df["team1.formation.lineup"]:
    for p in player_list:
        all_owngoal_values.append(p["ownGoals"])

pd.Series(all_owngoal_values).value_counts()
df.dtypes


# Artık işlevi kalmayan ham ve sabit sütunları temizle
drop_cols = [
    # Ham formasyon listeleri açtığımız sözlükler
    'team1.formation.bench','team1.formation.lineup','team1.formation.substitutions',
    'team2.formation.bench','team2.formation.lineup',"team1.coachId","team2_lineup_count",
    'team2.formation.substitutions',"team2.coachId",'referees',"team1_lineup_count",]

df.drop(columns=drop_cols, axis=1, inplace=True)

#######################################################
# Label değişkeni saçma scor bilgis yanlış sil
#######################################################
df['match_name'] = df['label'].str.split(',').str[0].str.replace(' - ', ' vs ')
df.drop(columns=['label'], inplace=True)

########################################################
# Değişken Tiplerini Düzelt
########################################################
# Daikaları float dan int e çeviri
sub_minute = df.columns[df.columns.str.contains("_sub_minute")]
df[sub_minute] = df[sub_minute].astype("Int64")

# Tarihleri düzelt +2 genle saat için
df["dateutc"] = pd.to_datetime(df["dateutc"])
df["dateutc"] = df["dateutc"] + pd.Timedelta(hours=2)

#######################################################
# Veri seti değişkenlerin düzenii
#######################################################
ordered_cols = [
    # Maç Kimliği ve Takvim
    'wyId', 'competitionId', 'gameweek', 'duration', 'dateutc', 'match_name', 'venue', 'winner',
    # Takım 1
    'team1.teamId', 'team1.side', 'team1.score', 'team1.scoreHT',
    # Takım2
    'team2.teamId', 'team2.side', 'team2.score', 'team2.scoreHT',
    # Kadro ve Disiplin Metrikleri
    'team1_bench_count', 'team2_bench_count','team1_yellow_cards', 'team1_red_cards',
    'team2_yellow_cards', 'team2_red_cards','team1_own_goals', 'team2_own_goals',
    # Değişiklik Zamanları
    'team1_sub_count', 'team2_sub_count',
    'team1_first_sub_minute', 'team1_last_sub_minute',
    'team2_first_sub_minute', 'team2_last_sub_minute']


df_new = df[ordered_cols]
# eda kısmında gördüm 18 tane kendi kalsine şut diyor bu yüzden sildik
df_new = df.drop(["team1_own_goals","team2_own_goals"],axis=1)
df_new.to_csv("matches_eng.csv",index=False)


#######################################
# Fonksiyon
#######################################
def matches_eng(dataframe,eutc):
    """ eut değeri ilgili data sette ne is o girilcek"""
    import pandas as pd
    import numpy as np
    import ast

    ####################################################
    # 2'den az eşsiz değerli sütunları sil #
    ####################################################
    drop_cols = dataframe.nunique()[dataframe.nunique() < 2].index
    # duration değişkeni uzatma devresine gidilmişmi bilgisi veriyor,
    # competitionId liglerin İd  sini veriyor bu yüzden silmiyoruz.
    drop_cols = [col for col in drop_cols if col not in ["duration", "competitionId"]]
   # geri kalanı siliyoruz
    dataframe.drop(drop_cols, axis=1, inplace=True)

    ###################################################
    # Sözlük yapılarını açma ve değişlenlere çevirme
    ###################################################
    # Nan değerl olduğu için sözlük yapısnı çözmede sıkıntı yaşadık bu yüzden fonsiyon yazdık
    def parse_formation(x):
        if isinstance(x, (list, dict)):
            return x
        if pd.isnull(x):
            return []
        x = x.replace("null", "None")
        return ast.literal_eval(x)

    # teamsData sütununu çevir
    dataframe["teamsData"] = dataframe["teamsData"].apply(parse_formation)
    # Hem team1 hem team2'nin formation ile ilgili TÜM sütunlarını bul ve çevir
    formation_cols = dataframe.columns[dataframe.columns.str.contains("formation", regex=False)]

    for col in formation_cols:
        dataframe[col] = dataframe[col].apply(parse_formation)

    # aynı bilgiyi içerdiği için sildik
    dataframe.drop(["team1.formation", "team2.formation"], axis=1, inplace=True)

    # gereksiz bilgimi içeriyor "teasmdata" seti
    dataframe.drop("teamsData", axis=1, inplace=True)
    # Kadro büyüklükleri (kaç kişi ilk 11'de, kaç kişi yedekte)
    dataframe["team1_lineup_count"] = dataframe["team1.formation.lineup"].apply(len)
    dataframe["team1_bench_count"] = dataframe["team1.formation.bench"].apply(len)
    dataframe["team2_lineup_count"] = dataframe["team2.formation.lineup"].apply(len)
    dataframe["team2_bench_count"] = dataframe["team2.formation.bench"].apply(len)
    # Kaç değişiklik yapıldı
    dataframe["team1_sub_count"] = dataframe["team1.formation.substitutions"].apply(len)
    dataframe["team2_sub_count"] = dataframe["team2.formation.substitutions"].apply(len)

    # Sarı / kırmızı kart SAYISI (değer '0' değilse, o oyuncu kart görmüş demektir - SAYIYORUZ, TOPLAMIYORUZ)
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

    # Kendi kalesine gol SAYISI (aynı mantık - '0' değilse gerçekleşmiş demektir)
    dataframe["team1_own_goals"] = dataframe["team1.formation.lineup"].apply(lambda x: count_events(x, "ownGoals")) \
                            + dataframe["team1.formation.bench"].apply(lambda x: count_events(x, "ownGoals"))
    dataframe["team2_own_goals"] = dataframe["team2.formation.lineup"].apply(lambda x: count_events(x, "ownGoals")) \
                            + dataframe["team2.formation.bench"].apply(lambda x: count_events(x, "ownGoals"))

    # İlk / son değişiklik dakikası
    dataframe["team1_first_sub_minute"] = dataframe["team1.formation.substitutions"].apply(
        lambda x: min(s["minute"] for s in x) if len(x) > 0 else np.nan)
    dataframe["team1_last_sub_minute"] = dataframe["team1.formation.substitutions"].apply(
        lambda x: max(s["minute"] for s in x) if len(x) > 0 else np.nan)

    dataframe["team2_first_sub_minute"] = dataframe["team2.formation.substitutions"].apply(
        lambda x: min(s["minute"] for s in x) if len(x) > 0 else np.nan)
    dataframe["team2_last_sub_minute"] = dataframe["team2.formation.substitutions"].apply(
        lambda x: max(s["minute"] for s in x) if len(x) > 0 else np.nan)
    dataframe.head()

    # Artık işlevi kalmayan ham ve sabit sütunları temizle
    drop_cols = [
        # Ham formasyon listeleri açtığımız sözlükler
        'team1.formation.bench', 'team1.formation.lineup', 'team1.formation.substitutions',
        'team2.formation.bench', 'team2.formation.lineup', "team1.coachId", "team2_lineup_count",
        'team2.formation.substitutions', "team2.coachId", 'referees', "team1_lineup_count", ]

    dataframe.drop(columns=drop_cols, axis=1, inplace=True)

    #######################################################
    # Label değişkeni saçma scor bilgis yanlış sil
    #######################################################
    dataframe['match_name'] = dataframe['label'].str.split(',').str[0].str.replace(' - ', ' vs ')
    dataframe.drop(columns=['label'], inplace=True)

    ########################################################
    # Değişken Tiplerini Düzelt
    ########################################################
    # Daikaları float dan int e çeviri
    sub_minute = dataframe.columns[dataframe.columns.str.contains("_sub_minute")]
    dataframe[sub_minute] = dataframe[sub_minute].astype("Int64")
    # Tarihleri düzelt +2 genle saat için
    dataframe["dateutc"] = pd.to_datetime(dataframe["dateutc"])
    dataframe["dateutc"] = dataframe["dateutc"] + pd.Timedelta(hours=eutc)

    #######################################################
    # Veri seti değişkenlerin düzenii
    #######################################################
    ordered_cols = [
        # Maç Kimliği ve Takvim
        'wyId', 'competitionId', 'gameweek', 'duration', 'dateutc', 'match_name', 'venue', 'winner',
        # Takım 1
        'team1.teamId', 'team1.side', 'team1.score', 'team1.scoreHT',
        # Takım2
        'team2.teamId', 'team2.side', 'team2.score', 'team2.scoreHT',
        # Kadro ve Disiplin Metrikleri
        'team1_bench_count', 'team2_bench_count', 'team1_yellow_cards', 'team1_red_cards',
        'team2_yellow_cards', 'team2_red_cards', 'team1_own_goals', 'team2_own_goals',
        # Değişiklik Zamanları
        'team1_sub_count', 'team2_sub_count',
        'team1_first_sub_minute', 'team1_last_sub_minute',
        'team2_first_sub_minute', 'team2_last_sub_minute']

    df_new = dataframe[ordered_cols]
    df_new = df_new.drop(["team1_own_goals", "team2_own_goals"], axis=1)
    df_new.to_csv("matches_engg.csv", index=False)
    return df_new

df_new= matches_eng(df,+2)

















