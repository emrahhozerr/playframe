import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

norm = pd.read_csv(r"C:\Users\ahmet\PycharmProjects\Miull Data Sciences Final Project\5-Pipline\data\team_season_tactics_normalized.csv")

# feature: sadece z-score kolonlari (oyuncu tarafiyla ayni mantik)
feature_cols = [c for c in norm.columns if c.endswith("_z")]
X = norm[feature_cols].fillna(0)

# oyuncudan farkli olarak burada pozisyon grubu yok - butun takimlar
# tek grup olarak kumeleniyor, lig kucuk oldugu icin (20 takim) az sayida
# kume yeterli
n_clusters = 3
km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
norm["style_cluster"] = km.fit_predict(X)


# --- KUME SAYISI (n_clusters=3) DOGRULAMASI: elbow + silhouette score ---
# yukaridaki X'i (ayni takimlar, ayni feature'lar) kullanir, yukarida
# hesaplanan style_cluster'i DEGISTIRMEZ - sadece n_clusters=3'un
# istatistiksel olarak makul olup olmadigini raporlar.
K_ARALIGI = range(2, 9)
MEVCUT_K = n_clusters  # 3
n_takim = len(X)

print(f"\n== Taktik Motoru (n={n_takim} takim) ==")
sonuclar = []
for k in K_ARALIGI:
    if n_takim < k * 2:
        continue
    km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiketler = km_test.fit_predict(X)
    sil = silhouette_score(X, etiketler) if len(set(etiketler)) > 1 else float("nan")
    sonuclar.append({"k": k, "inertia": km_test.inertia_, "silhouette": sil})

df_sonuc = pd.DataFrame(sonuclar)
print(df_sonuc.to_string(index=False))

if df_sonuc["silhouette"].notna().any():
    en_iyi_k = df_sonuc.loc[df_sonuc["silhouette"].idxmax(), "k"]
    print(f"  onerilen k={int(en_iyi_k)}  (su an kullanilan: {MEVCUT_K})")


def similar_team(team_id, top_n=5):
    idx = norm[norm["teamId"] == team_id].index[0]
    sims = cosine_similarity(X.loc[[idx]], X)[0]
    result = norm[["teamId", "style_cluster"]].copy()
    result["similarity"] = sims
    result = result[result["teamId"] != team_id]
    result = result.sort_values("similarity", ascending=False).head(top_n)
    return result

similar_team(1611)

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