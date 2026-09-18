"""
query_runner.py
--------
full_pipeline.py bir kere calisip "data/" klasorunu doldurduktan sonra,
tek bir oyuncu/takim/mac icin scout/taktik/radar/pas agi sorgusu yapmak
istedigin zaman BU dosyayi kullan. Boylece her sorgu icin butun 3 asamayi
(Data Engineer + Preprocessing + Feature Engineer) yeniden calistirman
gerekmez - main()'in zaten "data/" klasorune kaydettigi CSV'leri direkt
okur, sonuc saniyeler icinde gelir.

Fonksiyonlar full_pipeline.py'dekilerle AYNI (import degil, kopya) -
full_pipeline.py'nin adi/numarasi degisse bile bu dosya bozulmasin diye.

KULLANIM: full_pipeline.py'nin main()'ini en az bir kere calistirmis olman
gerekiyor (data/ klasorunde player_season_normalized.csv,
team_season_tactics_normalized.csv ve eng_players_data_prep.csv olmali).
Bu dosyayi full_pipeline.py ile AYNI klasore koy, `python query_runner.py`
calistir - dosyayi acip kod degistirmene gerek yok, calisinca sana bir
menu gosterip hangi sorguyu (scout/radar/taktik/pas agi) yapmak
istedigini ve gerekli oyuncu adi/takim id/mac id'yi soruyor.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ==========================================================================
# AI YORUM KATMANI (Google Gemini) — her sorgu sonucunun altina, tecrubeli
# bir scout/analist agzindan 2-3 cumlelik teknik bir yorum ekler. API
# anahtari yoksa (GOOGLE_API_KEY ortam degiskeni) ya da paket kurulu
# degilse, yorum sessizce/bir kere uyararak atlanir - sorgunun kendisini
# ASLA bozmaz (tablo/grafik her zaman gosterilir, yorum sadece bir ek).
# ==========================================================================

try:
    # ESKI paket ("google-generativeai" / import google.generativeai) Google
    # tarafindan tamamen kullanimdan kaldirildi (end of life) - artik guncelleme/
    # bug-fix almiyor. Yeni resmi paket "google-genai" (import: from google
    # import genai), API'si de biraz farkli (asagida ona gore yazildi).
    from google import genai
    from google.genai import errors as _genai_errors
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    _genai_errors = None

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# NOT: Google zaman zaman model isimlerini API'den kaldiriyor/degistiriyor
# (ornegin "gemini-1.5-flash" bir sure sonra kaldirildi ve 404 "model
# bulunamadi" hatasi vermeye basladi). Sabit bir isme guvenmek yerine,
# hesapta O AN aktif olan modelleri Google'dan canli sorup icinden uygun
# birini otomatik seciyoruz - boylece Google modelleri yeniden adlandirsa/
# kaldirsa bile kod elle guncellenmeden calismaya devam eder. Bu da
# calismazsa, sabit bir yedek liste sirayla denenir.
_GEMINI_YEDEK_ADAYLAR = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-lite-latest"]
_gemini_client = None
_gemini_model_adi = None
_gemini_adaylar = None  # henuz denenmemis model adi kuyrugu
_gemini_uyari_basildi = False


def _get_gemini_client():
    """genai.Client nesnesini bir kere kurup cache'ler."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _gemini_client


def _gemini_model_adaylarini_getir(client):
    """Hesapta generateContent destekleyen aktif model adlarini canli ceker;
    basaramazsa sabit yedek listeyi doner. 'flash' iceren (hizli, ucretsiz
    kotaya uygun) modelleri, 'lite'/'preview' gibi daha kararsiz varyantlara
    gore one alir."""
    try:
        adaylar = []
        for m in client.models.list():
            ad = getattr(m, "name", "") or ""
            ad = ad.split("/")[-1]
            if not ad:
                continue
            # SDK surumune gore bu alanin adi degisebiliyor - bulabilirsek
            # generateContent desteklemeyenleri eleriz, bulamazsak (alan
            # yoksa) yine de adayi listeye ekleriz.
            yontemler = (
                getattr(m, "supported_actions", None)
                or getattr(m, "supported_generation_methods", None)
            )
            if yontemler and "generateContent" not in yontemler:
                continue
            adaylar.append(ad)

        def skor(ad):
            ad_l = ad.lower()
            if "flash" in ad_l and "lite" not in ad_l and "preview" not in ad_l:
                return 0
            if "flash" in ad_l:
                return 1
            return 2

        if adaylar:
            adaylar.sort(key=skor)
            return adaylar
    except Exception:
        pass
    return list(_GEMINI_YEDEK_ADAYLAR)


def _gemini_model_adi_getir():
    """
    Denenecek bir sonraki model adini dondurur. ai_yorum() bu adla
    generate_content() cagrisi 404/"not found" ile basarisiz olursa
    _gemini_model_basarisiz() cagirip bir sonraki adaya gecmemizi saglar -
    boylece tek bir gecersiz model adi butun ozelligi kilitlemez.
    """
    global _gemini_model_adi, _gemini_adaylar
    if _gemini_model_adi is not None:
        return _gemini_model_adi
    client = _get_gemini_client()
    if _gemini_adaylar is None:
        _gemini_adaylar = _gemini_model_adaylarini_getir(client)
    if not _gemini_adaylar:
        return None
    _gemini_model_adi = _gemini_adaylar.pop(0)
    return _gemini_model_adi


def _gemini_model_basarisiz(hata):
    """
    Su anki model adiyla generate_content() basarisiz oldugunda ai_yorum()
    tarafindan cagrilir. Hata "model bulunamadi/desteklenmiyor" turundeyse
    (404 vb.) bu adi terk edip bir sonraki adaya gecmemizi saglar; baska
    turden bir hataysa (kota, ag, gecersiz key vb.) tekrar denemenin faydasi
    olmaz, sadece None dondurulur.
    """
    global _gemini_model_adi
    kod = getattr(hata, "code", None)
    metin = str(hata).lower()
    model_sorunu = kod == 404 or "404" in metin or "not found" in metin or "not supported" in metin
    if model_sorunu:
        _gemini_model_adi = None  # bir sonraki cagrida _gemini_adaylar'dan yeni bir isim denenecek
        return True
    return False


def ai_yorum(baglam):
    """
    Verilen "baglam" metnini (bir sorgunun sonuc tablosu/istatistikleri)
    Google Gemini'ye gonderip, tecrubeli bir futbol scoutu/teknik analist
    agzindan kisa (2-3 cumle), somut bir yorum ister - genel gecer lafla
    degil, verideki gercek sayi/isimlere atifla; eksik/zayif yonlere de
    deginmesi istenir. API kullanilamiyorsa ya da cagri basarisiz olursa
    None doner (cagiran kod bu durumda yorum bolumunu atlar).
    """
    global _gemini_uyari_basildi
    if not _GEMINI_AVAILABLE:
        if not _gemini_uyari_basildi:
            print("(AI yorumu icin 'pip install google-genai' gerekiyor - yorumlar atlanacak.)")
            _gemini_uyari_basildi = True
        return None
    if not GOOGLE_API_KEY:
        if not _gemini_uyari_basildi:
            print("(AI yorumu icin API key girilmedi - yorumlar atlanacak.")
            print(" Programi yeniden baslatip 'GOOGLE_API_KEY:' sorusuna ucretsiz bir anahtar "
                  "girerek (https://aistudio.google.com/app/apikey) bu ozelligi acabilirsin.)")
            _gemini_uyari_basildi = True
        return None

    prompt = (
        "Sen tecrubeli bir futbol scoutu ve teknik analistsin. Asagida "
        "bir veri analizi sorgusunun ciktisi var. Buna dayanarak KISA "
        "(2-3 cumle), somut ve teknik bir yorum yaz. Genel gecer, "
        "herkese uyan cumleler kurma - verideki gercek sayilara/"
        "isimlere/pozisyonlara atif yap. Guclu yanlarin yani sira "
        "eksik/zayif gorunen bir noktaya da deginmekten cekinme. "
        "Turkce yaz.\n\n---\n" + baglam
    )
    # ayni sorgu icinde en fazla birkac model adi denenir (biri "model
    # bulunamadi" derse digerine gecilir) - kullanicinin programi yeniden
    # baslatmasina gerek kalmasin diye
    for _ in range(4):
        model_adi = _gemini_model_adi_getir()
        if model_adi is None:
            if not _gemini_uyari_basildi:
                print("(AI yorumu icin uygun bir Gemini modeli bulunamadi - yorumlar atlanacak.)")
                _gemini_uyari_basildi = True
            return None
        try:
            client = _get_gemini_client()
            response = client.models.generate_content(model=model_adi, contents=prompt)
            return response.text.strip()
        except Exception as e:
            if _gemini_model_basarisiz(e):
                continue  # bir sonraki aday model adiyla tekrar dene
            print(f"(AI yorumu alinamadi: {e})")
            return None
    print("(AI yorumu alinamadi: denenen Gemini modellerinin hicbiri calismadi.)")
    return None


def ai_yorum_yazdir(baglam):
    """ai_yorum()'u cagirir, sonuc varsa basar - menu tarafinda tekrar eden 3 satiri kisaltmak icin."""
    yorum = ai_yorum(baglam)
    if yorum:
        print("\n[AI Teknik Yorum]")
        print(yorum)


def load_player_norm():
    """Scout Motoru / Oyuncu Gelisimi sorgulari icin oyuncu-sezon normalize verisini okur."""
    return pd.read_csv(os.path.join(DATA_DIR, "player_season_normalized.csv"))


def load_team_norm():
    """Taktik Motoru sorgulari icin takim-sezon normalize verisini okur."""
    return pd.read_csv(os.path.join(DATA_DIR, "team_season_tactics_normalized.csv"))


def load_events():
    """
    Pas agi sorgusu icin event bazli (mac ici aksiyon) veriyi okur.

    NOT: bu dosya cok buyuk (5 lig birlesik, milyonlarca satir) oldugu icin
    pandas onu parca parca okurken bazi ID kolonlarini (matchId/teamId/
    playerId) bazi satirlarda sayi, bazilarinda yazi (string) olarak
    algilayabiliyor ("DtypeWarning: mixed types" bunun isareti). Boyle
    olunca match_id/team_id ile filtreleme SESSIZCE bos sonuc donebiliyor
    (hata vermez, sadece hicbir satir bulamaz). Bunu onlemek icin
    low_memory=False ile tek seferde okuyup ID kolonlarini acikca sayiya
    ceviriyoruz.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "eng_players_data_prep.csv"), low_memory=False)
    for col in ["matchId", "teamId", "playerId"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_team_names():
    """
    Takim ID yerine takim ADI yazabilmek icin (Taktik Motoru / Pas Agi
    sorgularinda) kucuk "teamss.csv" dosyasini okur - butun event verisini
    (milyonlarca satir) sadece isim-ID eslemesi icin okumaya gerek yok.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "teamss.csv"))
    df = df.rename(columns={"wyId": "teamId"})
    return df[["teamId", "team_name"]]


# Kelime bazli arama SADECE takim adinin GECTIGI kelimeleri yakalar - "barca"
# gibi bir takma ad "Barcelona" icinde bitisik bir alt-dize olmadigi icin
# (b-a-r-c-a, "barcelona"nin icinde ardisik gecmiyor) bulunamaz. Bu 5 ligde
# (Ingiltere/Ispanya/Italya/Almanya/Fransa) yaygin kullanilan birkac takma
# adi, aramadan once "gercek" isme cevirmek icin kucuk bir sozluk - eksiksiz
# degil, ama en sik karsilasilan durumlari kapsiyor. Burada olmayan bir takma
# ad denenirse yine de kelime bazli arama devreye girer.
TEAM_NICKNAMES = {
    "barca": "barcelona",
    "atleti": "atletico madrid",
    "depor": "deportivo",
    "juve": "juventus",
    "inter": "internazionale",
    "psg": "paris saint germain",
    "om": "marseille",
    "ol": "lyon",
    "bvb": "borussia dortmund",
    "man utd": "manchester united",
    "man united": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham",
}


def resolve_team_id(query, team_names_df, valid_ids=None):
    """
    Takim ADINDAN teamId bulur - scout_engine_func'taki oyuncu-adi aramasiyla
    AYNI mantik: once tam eslesme (buyuk/kucuk harf onemsiz), yoksa kelime
    bazli kismi eslesme (yazdigin her kelime, sira/aralarinda baska kelime
    olmasi onemli degil, tam adin icinde geciyor mu). "Takim ID kimse bilmez,
    isim yazmak daha mantikli" geri bildirimi uzerine eklendi.

    Kelime bazli arama da bulamazsa, TEAM_NICKNAMES sozlugunde ("barca",
    "psg", "juve" gibi yaygin takma adlar) bir karsiligi var mi diye bakar
    ve onunla tekrar dener - cunku bir takma ad, resmi ismin icinde bitisik
    gecmeyebilir (ornek: "barca" -> "Barcelona"de ardisik degil).

    valid_ids verilirse (ornegin sadece elimizdeki tactic/event verisinde
    gecen takimlar) arama sadece o ID'ler arasinda yapilir, boylece Wyscout'un
    global teams.csv'sindeki (5 ligle alakasiz, binlerce) takimlar aday
    listesine girmez.

    Donus: 0 eslesme ya da >1 eslesme -> None (mesaj zaten basildi).
           tek eslesme -> (teamId, bulunan_team_name).
    """
    df = team_names_df.dropna(subset=["team_name"]).copy()
    if valid_ids is not None:
        df = df[df["teamId"].isin(valid_ids)]
    df = df.drop_duplicates(subset=["teamId"])

    query_clean = query.strip()
    query_lower = query_clean.lower()

    exact = df[df["team_name"].str.lower() == query_lower]
    if len(exact) >= 1:
        row = exact.iloc[0]
        return row["teamId"], row["team_name"]

    def word_search(text):
        words = [w for w in text.lower().split() if w]

        def all_words_match(name):
            return all(w in str(name).lower() for w in words)

        result = df[df["team_name"].apply(all_words_match)]
        return result.drop_duplicates(subset=["team_name"])

    aday = word_search(query_lower)

    if len(aday) == 0 and query_lower in TEAM_NICKNAMES:
        aday = word_search(TEAM_NICKNAMES[query_lower])

    if len(aday) == 0:
        print(f"'{query_clean}' adinda bir takim bulunamadi.")
        print("Ipucu: '5) LIGE GORE LISTELE' ile gecerli takim isimlerini gorebilirsin.")
        print("Ipucu: takma ad yerine resmi ismin bir parcasini (ornek: 'Barcelona') denemek de isabetli olabilir.")
        return None
    elif len(aday) > 1:
        print(f"'{query_clean}' tam eslesmedi, birden fazla aday bulundu:")
        for _, r in aday.head(15).iterrows():
            print("  -", r["team_name"])
        print("Lutfen tam ismi (yukaridakilerden birini, aynen kopyalayarak) tekrar dene.")
        return None
    else:
        row = aday.iloc[0]
        print(f"Not: tam eslesme yoktu, en yakin aday kullanildi -> {row['team_name']}")
        return row["teamId"], row["team_name"]


def resolve_player_name(query, norm):
    """
    Oyuncu ADINDAN tam player_name bulur - scout_engine_func icindeki
    similar_players() ile AYNI mantik (Wyscout oyunculari resmi tam adiyla,
    genelde gobek adi dahil, kayitli oldugu icin "Lionel Messi" gibi kisa
    bir arama tam eslesmeyebilir): once tam eslesme (buyuk/kucuk harf
    onemsiz), yoksa kelime bazli kismi eslesme (yazdigin her kelime, sira/
    aralarinda baska kelime olmasi onemli degil, tam adin icinde geciyor mu).
    OYUNCU GELISIMI (player_radar_func) icin eklendi, cunku orada Scout
    Motoru'ndaki AYNI "tam ad bulunamiyor" sorunu yasaniyordu.

    Donus: 0 ya da >1 eslesme -> None (mesaj zaten basildi).
           tek eslesme -> tam player_name (str).
    """
    isimler = norm["player_name"].dropna().drop_duplicates()
    query_clean = query.strip()
    query_lower = query_clean.lower()

    exact = isimler[isimler.str.lower() == query_lower]
    if len(exact) >= 1:
        return exact.iloc[0]

    query_words = [w for w in query_lower.split() if w]

    def all_words_match(name):
        return all(w in str(name).lower() for w in query_words)

    aday = isimler[isimler.apply(all_words_match)]

    if len(aday) == 0:
        print(f"'{query_clean}' bulunamadi (tam ya da kismi eslesme yok).")
        print("Ipucu: '5) LIGE GORE LISTELE' ile gecerli oyuncu isimlerini gorebilirsin.")
        print("Ipucu: sadece soyadini (ornek: sadece 'Messi') denemek de isabetli olabilir.")
        return None
    elif len(aday) > 1:
        print(f"'{query_clean}' tam eslesmedi, birden fazla aday bulundu:")
        for ad in aday.head(15):
            print("  -", ad)
        print("Lutfen tam ismi (yukaridakilerden birini, aynen kopyalayarak) tekrar dene.")
        return None
    else:
        found = aday.iloc[0]
        print(f"Not: tam eslesme yoktu, en yakin aday kullanildi -> {found}")
        return found


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
            # tam eslesme yok - Wyscout'ta oyuncular resmi tam adiyla (genelde
            # gobek adi dahil) kayitli oluyor, ornek: "Lionel Messi" degil
            # "Lionel Andres Messi Cuccittini" gibi. Bitisik substring arama
            # bunu yakalayamaz ("Andres" araya girdigi icin), o yuzden
            # KELIME BAZLI arama yapiyoruz: yazdigin her kelime (sira/arada
            # baska kelime olmasi onemli degil) tam ismin icinde geciyor mu
            query_words = [w for w in player_name.lower().split() if w]

            def all_words_match(name):
                name_lower = str(name).lower()
                return all(w in name_lower for w in query_words)

            aday = norm[norm["player_name"].apply(all_words_match)]
            aday_isimler = aday["player_name"].drop_duplicates().tolist()

            if len(aday_isimler) == 0:
                print(f"'{player_name}' bulunamadi (tam ya da kismi eslesme yok).")
                print("Ipucu: '5) LIGE GORE LISTELE' ile gecerli oyuncu isimlerini gorebilirsin.")
                print("Ipucu: sadece soyadini (ornek: sadece 'Messi') denemek de isabetli olabilir.")
                return None
            elif len(aday_isimler) > 1:
                print(f"'{player_name}' tam eslesmedi, birden fazla aday bulundu:")
                for ad in aday_isimler[:15]:
                    print("  -", ad)
                print("Lutfen tam ismi (yukaridakilerden birini, aynen kopyalayarak) tekrar dene.")
                return None
            else:
                player_name = aday_isimler[0]
                print(f"Not: tam eslesme yoktu, en yakin aday kullanildi -> {player_name}")
                matched = norm[norm["player_name"] == player_name]

        target = matched.iloc[0]
        role = target["role_code"]
        target_vector = X_all.loc[[target.name]]

        candidate_pool = pool[pool["role_code"] == role]
        candidate_vectors = pool_X.loc[candidate_pool.index]

        similarities = cosine_similarity(target_vector, candidate_vectors)[0]

        result_cols = ["player_name", "role_code", "style_cluster", "minutes_played"]
        if "league_code" in candidate_pool.columns:
            result_cols.insert(2, "league_code")  # hangi ligden oldugu gorunsun
        result = candidate_pool[result_cols].copy()
        result["similarity"] = similarities
        result = result[result["player_name"] != player_name]
        result = result.sort_values("similarity", ascending=False).head(top_n)
        return result

    scout_user = similar_players(user, top_n=head)
    if scout_user is None:
        return None  # oyuncu bulunamadi/belirsizdi, hata mesaji zaten basildi

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
        result_cols = ["teamId", "style_cluster"]
        if "league_code" in norm.columns:
            result_cols.insert(1, "league_code")  # hangi ligden oldugu gorunsun
        result = norm[result_cols].copy()
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
        print("kaydedildi:", out_path)
        plt.show()  # pencerede ac - kapatana kadar bekler
        plt.close(fig)
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
        print("kaydedildi:", out_path)
        plt.show()  # pencerede ac - kapatana kadar bekler
        plt.close(fig)
        return out_path

    if isinstance(player_names, str):
        player_names = [player_names]

    # scout_engine_func'ta yasadigimiz AYNI sorun buradaydi da: "Lionel Messi"
    # gibi kisa bir isim, Wyscout'un kayitli tuttugu resmi tam adla (gobek
    # adi dahil) tam eslesmiyor. Once her ismi tam player_name'e cozuyoruz,
    # bulunamayan/belirsiz olanlari (mesaji zaten basildi) listeden cikarip
    # devam ediyoruz - boylece ne yanlis "oyuncu bulunamadi" ile durup
    # kalmiyor ne de yanlislikla None ile devam edip hata veriyor.
    resolved_names = []
    for name in player_names:
        found = resolve_player_name(name, norm)
        if found is not None:
            resolved_names.append(found)
    player_names = resolved_names

    if not player_names:
        return None, None, []

    radar_path = plot_radar(player_names)

    stat_path = None
    if len(player_names) == 1:
        stat_path = full_stat_sheet(player_names[0], top_n)

    # ucuncu deger olarak COZULMUS (tam) isim listesini de donduruyoruz -
    # cagiran kod (menu) AI yorumu icin percentile verisini tekrar cozmeden
    # dogrudan bu isimlerle player_norm'a bakabilsin diye
    return radar_path, stat_path, player_names

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
    plt.show()  # pencerede ac - kapatana kadar bekler
    plt.close(fig)


def pas_agi_ozet(ev, match_id, team_id):
    """
    pass_network() ile AYNI filtre/hesap mantigini kullanarak (ama grafik
    cizmeden), o mac+takim icin sayisal bir OZET metni uretir - AI yoruma
    "baglam" olarak vermek icin. pass_network'un kendisine dokunmuyoruz
    (calisan gorsellestirmeyi bozma riski almamak icin), sadece ayni
    hesabi tekrar (hafif) yapiyoruz.
    """
    team_ev = ev[(ev["matchId"] == match_id) & (ev["teamId"] == team_id)].copy()
    team_ev = team_ev.sort_values(["matchPeriod", "eventSec"]).reset_index(drop=True)
    is_pass = team_ev["eventName"] == "Pass"
    is_pass_ok = is_pass & (team_ev["tag_1801"] == 1)

    team_ev["receiver_playerId"] = team_ev["playerId"].shift(-1)
    passes = team_ev[is_pass_ok].copy()
    passes = passes.dropna(subset=["receiver_playerId"])
    passes["receiver_playerId"] = passes["receiver_playerId"].astype(int)
    passes = passes[passes["playerId"] != passes["receiver_playerId"]]

    toplam_pas = int(is_pass.sum())
    isabetli_pas = int(is_pass_ok.sum())
    isabet_pct = (isabetli_pas / toplam_pas * 100) if toplam_pas > 0 else 0

    pair_counts = {}
    for _, row in passes.iterrows():
        key = tuple(sorted([int(row["playerId"]), int(row["receiver_playerId"])]))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    en_iyi_ikili = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    pas_hacmi = passes["playerId"].value_counts().add(
        passes["receiver_playerId"].value_counts(), fill_value=0
    ).sort_values(ascending=False)
    en_hareketli = pas_hacmi.head(3)

    satirlar = [
        f"Toplam pas denemesi: {toplam_pas}, isabetli: {isabetli_pas} (%{isabet_pct:.1f} isabet)",
        "En cok pas alisverisi yapan ikili(ler) (playerId, pas sayisi): "
        + ", ".join(f"({a}-{b}: {c})" for (a, b), c in en_iyi_ikili),
        "En hareketli oyuncular (playerId, toplam pas hacmi): "
        + ", ".join(f"{pid}: {int(v)}" for pid, v in en_hareketli.items()),
    ]
    return "\n".join(satirlar)


# ==========================================================================
# MENU — dosyayi her calistirdiginda hangi sorguyu yapmak istedigini sorar,
# kod icinde satir acip/kapatmana gerek kalmasin diye
# ==========================================================================

def run_menu():
    print("Ne yapmak istersin?")
    print("1) SCOUT MOTORU      - oyuncuya en cok benzeyen oyunculari bul")
    print("2) OYUNCU GELISIMI   - radar grafik + tam istatistik ozeti (2-3 oyuncu karsilastirma da olur)")
    print("3) TAKTIK MOTORU     - takima en cok benzeyen takimlari bul")
    print("4) PAS AGI           - belirli mac+takim icin pas agi haritasi")
    print("5) LIGE GORE LISTELE - bir ligden ornek oyuncu adi / takim adi bul")
    print("0) Cikis")
    secim = input("Secimin (0-5): ").strip()

    if secim == "1":
        isim = input("Oyuncu adi (ornek: Eden Hazard): ").strip()
        head = input("Kac benzer oyuncu listelensin? (bos birak = 10): ").strip()
        head = int(head) if head else 10
        player_norm = load_player_norm()
        sonuc = scout_engine_func(player_norm, isim, head=head)
        if sonuc is not None:
            print(sonuc)
            baglam = (
                f"Sorgulanan oyuncu: {isim}\n"
                f"Ayni pozisyondan en benzer {head} oyuncu (isim, pozisyon, "
                f"[lig,] stil kumesi, dakika, benzerlik skoru):\n"
                f"{sonuc.to_string(index=False)}"
            )
            ai_yorum_yazdir(baglam)

    elif secim == "2":
        print("TEK oyuncu icin: sadece ismini yaz -> Eden Hazard")
        print("KARSILASTIRMA icin: isimleri virgulle ayirarak yaz -> Eden Hazard, Mohamed Salah")
        isimler = input("Oyuncu adi(lari): ").strip()
        isim_listesi = [i.strip() for i in isimler.split(",") if i.strip()]
        if len(isim_listesi) > 3:
            print("En fazla 3 oyuncu karsilastirabilirsin, ilk 3'u aliyorum.")
            isim_listesi = isim_listesi[:3]
        player_norm = load_player_norm()
        if len(isim_listesi) == 1:
            print(f"-> tek oyuncu modu: {isim_listesi[0]} (radar + tam istatistik ozeti)")
        else:
            print(f"-> karsilastirma modu: {', '.join(isim_listesi)} (sadece radar, ust uste)")
        # tek oyuncuysa liste yerine string de gecilebilir (player_radar_func ikisini de kabul eder)
        _, _, cozulen_isimler = player_radar_func(
            player_norm, isim_listesi if len(isim_listesi) > 1 else isim_listesi[0]
        )
        if cozulen_isimler:
            pct_cols = [c for c in player_norm.columns if c.endswith("_p90_percentile")]
            satirlar = []
            for ad in cozulen_isimler:
                row = player_norm.loc[player_norm["player_name"] == ad].iloc[0]
                pct = row[pct_cols].dropna().astype(float).sort_values()
                if pct.empty:
                    continue
                en_dusuk = pct.head(3)
                en_yuksek = pct.tail(3)
                satirlar.append(
                    f"{ad} ({row['role_code']}): en guclu -> "
                    + ", ".join(f"{m.replace('_p90_percentile', '')}: %{v:.0f}" for m, v in en_yuksek.items())
                    + " | en zayif -> "
                    + ", ".join(f"{m.replace('_p90_percentile', '')}: %{v:.0f}" for m, v in en_dusuk.items())
                )
            if satirlar:
                baglam = (
                    "Oyuncu(lar)in pozisyon-ici percentile siralamasi (0-100, "
                    "100 = ayni pozisyondaki oyuncular arasinda en iyi):\n"
                    + "\n".join(satirlar)
                )
                ai_yorum_yazdir(baglam)

    elif secim == "3":
        isim = input("Takim adi (ornek: Chelsea): ").strip()
        top_n = input("Kac benzer takim listelensin? (bos birak = 5): ").strip()
        top_n = int(top_n) if top_n else 5
        team_norm = load_team_norm()
        team_names = load_team_names()
        resolved = resolve_team_id(isim, team_names, valid_ids=team_norm["teamId"].unique())
        if resolved is not None:
            team_id, bulunan_isim = resolved
            sonuc = tactic_cluster(team_norm, team_id=team_id, top_n=top_n, csv=True)
            sonuc = sonuc.merge(team_names, on="teamId", how="left")
            cols = ["teamId", "team_name"] + [c for c in sonuc.columns if c not in ("teamId", "team_name")]
            print(f"\n'{bulunan_isim}' takimina en benzer {top_n} takim:")
            print(sonuc[cols].to_string(index=False))
            baglam = (
                f"Sorgulanan takim: {bulunan_isim}\n"
                f"Taktik/stil acisindan en benzer {top_n} takim (isim, "
                f"[lig,] stil kumesi, benzerlik skoru):\n"
                f"{sonuc[cols].to_string(index=False)}"
            )
            ai_yorum_yazdir(baglam)

    elif secim == "4":
        isim = input("Pas agini gormek istedigin takimin adi (ornek: Chelsea): ").strip()
        events = load_events()
        team_names = load_team_names()
        resolved = resolve_team_id(isim, team_names, valid_ids=events["teamId"].dropna().unique())
        if resolved is not None:
            team_id, bulunan_isim = resolved

            takim_ev = events[events["teamId"] == team_id]
            # NOT: matches_engg.csv'deki "match_name" kolonu, event verisiyle
            # birlestirilirken TEKRAR "match_" onekini aliyor -> "match_match_name"
            name_col = "match_match_name" if "match_match_name" in takim_ev.columns else "match_name"
            mac_cols = [c for c in ["matchId", name_col, "match_dateutc"] if c in takim_ev.columns]
            maclar = takim_ev[mac_cols].drop_duplicates(subset=["matchId"])
            if "match_dateutc" in maclar.columns:
                maclar = maclar.sort_values("match_dateutc")
            maclar = maclar.reset_index(drop=True)

            if maclar.empty:
                print(f"'{bulunan_isim}' icin hicbir mac/event bulunamadi.")
            else:
                print(f"\n'{bulunan_isim}' takiminin maclari:")
                for i, row in maclar.iterrows():
                    tarih = row["match_dateutc"] if "match_dateutc" in maclar.columns else ""
                    ad = row[name_col] if name_col in maclar.columns else f"matchId={row['matchId']}"
                    print(f"  {i + 1}) {tarih}  {ad}")

                secim_no = input(f"Hangi mac? (1-{len(maclar)}): ").strip()
                try:
                    secim_no = int(secim_no)
                except ValueError:
                    secim_no = -1

                if not (1 <= secim_no <= len(maclar)):
                    print("Gecersiz secim.")
                else:
                    match_id = int(maclar.loc[secim_no - 1, "matchId"])
                    n_found = ((events["matchId"] == match_id) & (events["teamId"] == team_id)).sum()
                    if n_found == 0:
                        print("UYARI: bu mac icin hicbir event bulunamadi (beklenmedik durum).")
                    else:
                        print(f"{n_found} event bulundu, pas agi ciziliyor...")
                        pass_network(events, match_id=match_id, team_id=team_id)
                        print("kaydedildi: pass_network.png")
                        ozet = pas_agi_ozet(events, match_id=match_id, team_id=team_id)
                        baglam = (
                            f"Takim: {bulunan_isim}, matchId={match_id}\n"
                            f"Pas agi ozet istatistikleri:\n{ozet}"
                        )
                        ai_yorum_yazdir(baglam)

    elif secim == "5":
        player_norm = load_player_norm()
        if "league_code" not in player_norm.columns:
            print("Bu veri setinde league_code kolonu yok (tek ligli calisilmis).")
        else:
            ligler = sorted(player_norm["league_code"].dropna().unique().tolist())
            print("Mevcut ligler:", ", ".join(ligler))
            lig = input("Hangi lig? (ornek: Spain): ").strip()
            if lig not in ligler:
                print(f"'{lig}' bulunamadi. Gecerli ligler: {ligler}")
            else:
                secim2 = input("Oyuncu / Takim listelensin? (o/t): ").strip().lower()
                if secim2 == "t":
                    team_norm = load_team_norm()
                    team_names = load_team_names()
                    ornekler = team_norm.loc[team_norm["league_code"] == lig, ["teamId"]].drop_duplicates()
                    ornekler = ornekler.merge(team_names, on="teamId", how="left")
                    print(f"{lig} ligindeki takimlar ({len(ornekler)} takim):")
                    print(ornekler[["team_name", "teamId"]].to_string(index=False))
                    print("\nBu isimlerden birini 'TAKTIK MOTORU' (3) ya da 'PAS AGI' (4) sorgusunda kullanabilirsin.")

                else:
                    ornekler = player_norm.loc[player_norm["league_code"] == lig,
                                                ["player_name", "role_code"]].drop_duplicates().head(20)
                    print(f"{lig} ligindeki oyunculardan ornekler (ilk 20):")
                    print(ornekler.to_string(index=False))

    elif secim == "0":
        print("Cikildi.")
        return False

    else:
        print("Gecersiz secim, 0-5 arasi bir sayi gir.")

    return True


if __name__ == "__main__":
    os.chdir(DATA_DIR)  # png/csv ciktilari data/ klasorune yazilsin

    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
    pd.set_option("expand_frame_repr", False)
    pd.set_option("float_format", lambda x: "%.3f" % x)

    # Program baslarken (menuden once) Google Gemini API key'ini soruyoruz -
    # boylece her sorgu sonrasi otomatik "[AI Teknik Yorum]" eklenebiliyor.
    # Bos birakilirsa AI yorum katmani sessizce devre disi kalir, program
    # normal calismaya devam eder (sorgu/tablo/grafik ciktisi hic etkilenmez).
    if not GOOGLE_API_KEY:
        print("Google Gemini API key'ini gir (yorumlar icin, bos gecebilirsin):")
        print("  -> ucretsiz key: https://aistudio.google.com/app/apikey")
        girilen_key = input("GOOGLE_API_KEY: ").strip()
        if girilen_key:
            GOOGLE_API_KEY = girilen_key
        print()

    # her sorgudan sonra tekrar menu gostersin, "0" ile cikana kadar
    devam = True
    while devam:
        devam = run_menu()
        if devam:
            tekrar = input("\nBaska bir sorgu yapmak ister misin? (e/h): ").strip().lower()
            devam = tekrar == "e"