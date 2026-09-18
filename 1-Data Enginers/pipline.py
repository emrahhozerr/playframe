

"""
main.py
--------
Tüm veri işleme pipeline'larını doğru sırada çalıştırır.

NEDEN BU SIRA?
`all_dataframes_pipline`, diğer fonksiyonların diske yazdığı ara CSV
dosyalarını (playerss.csv, teamss.csv, players_gamee.csv,
event_id_2_namee.csv, tags_2_namee.csv) okuyarak birleştirme yapıyor.
Bu yüzden önce o dosyaları üreten fonksiyonlar, en son da
`all_dataframes_pipline` çalıştırılmalı.

KULLANIM:
Aşağıdaki "GİRİŞ DOSYALARI" bölümündeki yolları kendi dosya
konumlarınla değiştir, sonra `python main.py` çalıştır.
"""

def players_pipline(players):
    """
    BİLGİ ="foot","currentTeamId","currentNationalTeamId" bu üç değişkenin
    eksik değişrleri "unknow" ile doldurdu
    """
    import pandas as pd
    import numpy as np
    import ast
    dataframe = pd.read_csv(players)
    # Sözlük Halindeki değişkenleri düzelt #

    # String halindeki dict'i gerçek dict'e çevir
    dataframe["role"] = dataframe["role"].apply(ast.literal_eval)
    dataframe["passportArea"] = dataframe["passportArea"].apply(ast.literal_eval)
    dataframe["birthArea"] = dataframe["birthArea"].apply(ast.literal_eval)

    # role sütunundan sadece pozisyon kodunu çıkar (GK, dataframe, MD, FW gibi)
    dataframe["role_code"] = dataframe["role"].apply(lambda x: x["code2"])
    dataframe["role_name"] = dataframe["role"].apply(lambda x: x["name"])

    # passportArea'dan ülke adını çıkar
    dataframe["passport_country"] = dataframe["passportArea"].apply(lambda x: x["name"])
    dataframe["passport_country_id"] = dataframe["passportArea"].apply(lambda x: x['id'])

    # birthArea'dan doğum yeri ülkesini çıkar
    dataframe["birth_country"] = dataframe["birthArea"].apply(lambda x: x["name"])
    dataframe["birth_country_id"] = dataframe["birthArea"].apply(lambda x: x["id"])

    # Sözlük yapılarından değişken üretiğimiz için  artık bu değişken
    # ile işimiz olmadığı için sildim
    dataframe.drop(["role", "passportArea", "birthArea"], axis=1, inplace=True)
    ###########################################################################
    # Aykırı Değerler

    # 0 olması gerek değerlere i nan yap
    dataframe["weight"] = dataframe["weight"].replace(0, np.nan)
    dataframe["height"] = dataframe["height"].replace(0, np.nan)
    ##########################################################################
    # Nan Değerler #

    # Milli takımıda boşluklaru "unknow" yaptım oyuncu mili takıma seçilmemiş
    dataframe["currentNationalTeamId"] = dataframe["currentNationalTeamId"].fillna("unknow")

    # takım id sinide unkow yapcam belki oyuncu o yıl takımsız kaldı yada genç oyuncu
    dataframe["currentTeamId"] = dataframe["currentTeamId"].fillna("unknow")

    # BOY VE kiloda o mevkiye göre median alıp oan göre doldurma yapcam median
    # olmasının sebebi sporcu verilerin normal bir davranış sergilememesinden dolayı
    # a) boy nan değerler
    dataframe['height'] = dataframe.groupby('role_code')['height'].transform(
        lambda x: x.fillna(x.median()))

    # b) kilo nan değeler
    dataframe['weight'] = dataframe.groupby('role_code')['weight'].transform(
        lambda x: x.fillna(x.median()))

    # hangi ayağı kullandğı bilgi eksik kişiler var "unknow" olarak
    # değiştirildi
    dataframe["foot"] = dataframe["foot"].fillna("unknow")
    ##########################################################################
    # Tarih Verilerinin Tipini Düzeltme #
    dataframe["birthDate"] = pd.to_datetime(dataframe["birthDate"])
    ##########################################################################
    # Gereksiz Değişkenlerden Kurtulma,Oluşturma,Düzeltme #

    # İsim ve Soy isimleri birleştir
    dataframe["name"] = dataframe["firstName"] + " " + dataframe["lastName"]
    dataframe.drop(["firstName", "lastName"], axis=1, inplace=True)

    # isimlerdeki bozukluğu gider
    dataframe["name"] = dataframe["name"].apply(lambda x: x.encode().decode("unicode_escape"))
    dataframe["shortName"] = dataframe["shortName"].apply(lambda x: x.encode().decode("unicode_escape"))

    # oyuncu adını "player_name olacak sonra yap"
    dataframe.rename(columns={"name": "player_name", }, inplace=True)
    #############################################################################
    # Data Frame Sırlama #
    # Profesyonel ve mantıksal sütun sıralaması
    ordered_columns = [
        # 1. Kimlik
        "wyId", "player_name", "shortName",
        # 2. Rol ve Takım
        "role_code", "role_name", "currentTeamId",
        # 3. Fiziksel ve Biyolojik
        "birthDate", "height", "weight", "foot",
        # 4. Ülke Bilgileri
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
    # name yazıyor karışıklık olmasın diye team name yaptım oyuncu adında name yazıyor
    dataframe.rename(columns={"name": "team_name"}, inplace=True)

    # ingilizce formata gelidği için diğer dillerden geçerkken
    # kelime yapaısnı bozmuş onu düzgün düzeltelim
    dataframe["team_name"] = dataframe["team_name"].apply(lambda x: x.encode().decode("unicode_escape"))
    dataframe["officialName"] = dataframe["officialName"].apply(lambda x: x.encode().decode("unicode_escape"))
    dataframe["city"] = dataframe["city"].apply(lambda x: x.encode().decode("unicode_escape"))

    # area değişkeni sözlük yapısı olarak tam onumamış onu düzeltelim
    dataframe["area"] = dataframe["area"].apply(ast.literal_eval)  #

    # area sözlük yapısını değişkene veririyoruz
    dataframe["area_team_conutry"] = dataframe["area"].apply(lambda x: x['name'])
    dataframe["area_team_conutry_id"] = dataframe["area"].apply(lambda x: x['id'])
    dataframe.drop("area", axis=1, inplace=True)

    # Columns Sırasını düzenleme
    ordered_team_columns = ['wyId', 'team_name', 'officialName', 'type', 'city', 'area_team_conutry', 'area_team_conutry_id']
    # DÜZELTME: burada tanımsız "df" değil, fonksiyon içindeki "dataframe" kullanılmalı
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
    import ast
    dataframe = pd.read_csv(players_games)
    # Gereksiz Sütünlardan kurtul
    del_cols = ['Unnamed: 0', 'firstname', 'lastname', "jersey_number"]
    dataframe.drop(del_cols, axis=1, inplace=True)

    # shortName
    dataframe.rename(columns={"nickname": "shortName"}, inplace=True)

    # tarih değişkenin tipini düzelt
    dataframe["birth_date"] = pd.to_datetime(dataframe["birth_date"])

    # minutes_played değişkeninde 0 ve altı olan ifadeleri nan yap
    dataframe.loc[dataframe["minutes_played"] <= 0, "minutes_played"] = np.nan

    # 0 olan ları nan çevirmiştik şimdi onlaru kaldırıyorum
    dataframe.dropna(subset=["minutes_played"], axis=0, inplace=True)

    # Sıralamsı
    ordered_cols = [
        # 1. Anahtarlar ve Eşleşme ID'leri
        'game_id', "team_id", "player_id",
        # 2. Oyuncu Tanımlayıcıları
        'player_name', 'shortName', 'birth_date',
        # 3. Maç İçi Metrikler / Durumlar
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

    # Artık işlevi kalmayan ham ve sabit sütunları temizle
    drop_cols = [
        # Ham formasyon listeleri açtığımız sözlükler
        'team1.formation.bench', 'team1.formation.lineup', 'team1.formation.substitutions',
        'team2.formation.bench', 'team2.formation.lineup', "team1.coachId", "team2_lineup_count",
        'team2.formation.substitutions', "team2.coachId", 'referees', "team1_lineup_count", ]

    # DÜZELTME: errors="ignore" eklendi. Bu sütunlardan biri veri setinde
    # yoksa (örn. "referees" bazı kaynaklarda bulunmuyor) eskiden KeyError
    # fırlatıyordu; artık var olan sütunlar sessizce silinir.
    dataframe.drop(columns=drop_cols, axis=1, inplace=True, errors="ignore")

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

def events_eng_pipline(events_eng):
    import pandas as pd
    import numpy as np
    import ast
    dataframe = pd.read_csv(events_eng)
    dataframe = dataframe.drop(columns=["pos_orig_x", "pos_orig_y", "pos_dest_x", "pos_dest_y"])

    # pos_orig_x/y, pos_dest_x/y kolonlarini positions'tan kendimiz dogru sekilde uretiyoruz
    pattern = r"\{'y':\s*(-?\d+),\s*'x':\s*(-?\d+)\}"
    matches = dataframe["positions"].str.findall(pattern)
    dataframe["pos_orig_y"] = matches.apply(lambda m: int(m[0][0]) if len(m) > 0 else np.nan)
    dataframe["pos_orig_x"] = matches.apply(lambda m: int(m[0][1]) if len(m) > 0 else np.nan)
    dataframe["pos_dest_y"] = matches.apply(lambda m: int(m[1][0]) if len(m) > 1 else np.nan)
    dataframe["pos_dest_x"] = matches.apply(lambda m: int(m[1][1]) if len(m) > 1 else np.nan)

    # bilgi tekrarı değişkelerde
    dataframe.drop(["positions", "tags"], axis=1, inplace=True)

    # Tag kısmı ayır değişkenlere aldık bir kullanıcı bir den fazla tagı pozisyonu var
    # tag_list ıd olarak değiştir listeden çıkar
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

    # artık işimiz olmadığı için çıkardık
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
                                        right_on="match_wyId", how="left")
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
    long_tags = events_raw.melt(id_vars=["id"], value_vars=tag_cols,
                                var_name="tag_col", value_name="present")
    long_tags = long_tags[long_tags["present"] == 1]
    long_tags["Tag"] = long_tags["tag_col"].str.replace("tag_", "").astype(int)
    long_tags = long_tags.merge(tags_2_name, on="Tag", how="left")
    long_tags = long_tags.drop(columns=["tag_col", "present"])
    long_tags.to_csv("event_tags_long.csv", index=False)
    return players_game, event_england, long_tags
# ==========================================================================
# GİRİŞ DOSYALARI — kendi dosya yollarınla değiştir
# ==========================================================================
PLAYERS_CSV = r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data\players.csv"
TEAMS_CSV = r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data\teams.csv"
TAG2NAMES_CSV = r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data\tags2name.csv"
PLAYERS_GAMES_CSV = r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data\player_games.csv"
MATCHES_ENG_CSV = r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data\matches_England.csv"
EVENTS_ENG_CSV = r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data\events_England.csv"
EVENTID2NAME_CSV = r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\Data\eventid2name.csv"

# matches_eng_pipline içindeki dateutc sütununa eklenecek saat farkı
# (yorumda "+2 genle saat için" diye belirtilmiş, varsayılan olarak 2 aldım)
EUTC_OFFSET = 2


def main():
    print("1/8 - players_pipline çalışıyor...")
    players_pipline(PLAYERS_CSV)
    print("   -> playerss.csv oluşturuldu")

    print("2/8 - teams_pipline çalışıyor...")
    teams_pipline(TEAMS_CSV)
    print("   -> teamss.csv oluşturuldu")

    print("3/8 - tag2names_pipline çalışıyor...")
    tag2names_pipline(TAG2NAMES_CSV)
    print("   -> tags_2_namee.csv oluşturuldu")

    print("4/8 - players_games_pipline çalışıyor...")
    players_games_pipline(PLAYERS_GAMES_CSV)
    print("   -> players_gamee.csv oluşturuldu")

    print("5/8 - matches_eng_pipline çalışıyor...")
    matches_eng_pipline(MATCHES_ENG_CSV, EUTC_OFFSET)
    print("   -> matches_engg.csv oluşturuldu")

    print("6/8 - events_eng_pipline çalışıyor...")
    events_eng_pipline(EVENTS_ENG_CSV)
    print("   -> event_englandd.csv oluşturuldu")

    print("7/8 - eventid2_name_pipline çalışıyor...")
    eventid2_name_pipline(EVENTID2NAME_CSV)
    print("   -> event_id_2_namee.csv oluşturuldu")

    print("8/8 - all_dataframes_pipline çalışıyor (tüm tabloları birleştiriyor)...")
    # DİKKAT: burada matches_engg.csv (5. adımın çıktısı) ve
    # event_englandd.csv (6. adımın çıktısı) veriliyor; ham dosyalar değil.
    players_game, event_england, long_tags = all_dataframes_pipline(
        mat="matches_engg.csv",
        event="event_englandd.csv",
    )
    print("   -> player_game_merged.csv, event_england_merged.csv, event_tags_long.csv oluşturuldu")

    print("\nTüm pipeline tamamlandı.")
    print(f"players_game shape : {players_game.shape}")
    print(f"event_england shape: {event_england.shape}")
    print(f"long_tags shape    : {long_tags.shape}")


if __name__ == "__main__":
    main()