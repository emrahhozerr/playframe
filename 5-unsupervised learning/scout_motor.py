import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("expand_frame_repr", False)

norm = pd.read_csv(
    r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\5-Pipline\data\player_season_normalized.csv")
norm.shape

# feature seçimi sadece z-score kolonlarini kullaniyoruz
# z-score'lar zaten pozisyon grubuna role_code gore hesaplandigi icin
# farkli olcekteki metrikler orn. pas sayisi vs gol sayisi esit agirlikta
# kiyaslanabiliyor. yuzde/percentile/ham kolonlari feature'a katmiyoruz -
# ayni bilginin farkli bir gosterimi, cift sayim olur.
feature_cols = [col for col in norm.columns if col.endswith("_p90_z")]
len(feature_cols)

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

pool["role_code"].value_counts()
pool.groupby(["role_code", "style_cluster"]).size()


# --- KUME SAYISI (n_clusters=4) DOGRULAMASI: elbow + silhouette score ---
# yukaridaki pool/pool_X'i (ayni havuz, ayni feature'lar) kullanir, yukarida
# hesaplanan style_cluster'i DEGISTIRMEZ - sadece n_clusters=4'un pozisyon
# gruplarinda istatistiksel olarak makul olup olmadigini raporlar.
k_aralığı = range(2, 9)
mevcut_k = n_clusters  # 4

for role, idx in pool.groupby("role_code").groups.items():
    X_role = pool_X.loc[idx]
    n = len(X_role)
    print(f"\n== {role} (n={n}) ==")

    sonuclar = []

    for k in k_aralığı:
        if n < k * 2:
            continue
        km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        etiketler = km_test.fit_predict(X_role)
        sil = silhouette_score(X_role, etiketler) if len(set(etiketler)) > 1 else float("nan")
        sonuclar.append({"k": k, "inertia": km_test.inertia_, "silhouette": sil})

    df_sonuc = pd.DataFrame(sonuclar)
    if df_sonuc.empty:
        print("  (yeterli oyuncu yok, atlandi)")
        continue
    print(df_sonuc.to_string(index=False))

    if df_sonuc["silhouette"].notna().any():
        en_iyi_k = df_sonuc.loc[df_sonuc["silhouette"].idxmax(), "k"]
        print(f"  onerilen k={int(en_iyi_k)}  (su an kullanilan: {mevcut_k})")


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

similar_players("Eden Hazard", top_n=10).sort_values("similarity", ascending=False)

norm.to_csv("player_season_scouted.csv", index=False)

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