"""
dashboard.py
--------
Streamlit arayuzu - query_runner.py'nin AYNI mantigini (Scout Motoru,
Oyuncu Gelisimi/Radar, Taktik Motoru, Pas Agi, Lige Gore Listele, AI
Yorum) web arayuzune tasir. query_runner.py'ye DOKUNMAZ, sadece onun
okudugu "data/" klasorunu okur - ikisi yan yana durabilir.

KULLANIM:
1) full_pipeline / all_lig_full_pipeline en az bir kere calismis olmali
   (data/ klasorunde player_season_normalized.csv,
   team_season_tactics_normalized.csv, teamss.csv bulunmali).
2) Bu dosyayi 5-Pipline klasorune (query_runner.py ile AYNI klasore) koy.
3) Terminalde (conda ortaminda):
4) Acilan sayfada sol menuden istersen Google Gemini API key gir
   (bos birakilirsa AI yorum bolumu sessizce gizli kalir, sorgu/tablo/
   grafik sonuclari HICBIR ZAMAN bundan etkilenmez).
"""

import os
import io
import gzip
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

st.set_page_config(page_title="Futbol Analitik Dashboard", layout="wide")


# ==========================================================================
# AI YORUM KATMANI (Google Gemini) — query_runner.py'deki self-healing model
# secim mantigiyla AYNI, sadece API key'i global degil session bazli
# tutuyoruz (Streamlit'te ayni process birden fazla kullaniciya hizmet
# edebilir, key'ler karismasin diye api_key'e gore ayri client/model
# cache'liyoruz).
# ==========================================================================

try:
    from google import genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

_GEMINI_YEDEK_ADAYLAR = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-lite-latest"]
_gemini_clients = {}
_gemini_model_by_key = {}
_gemini_candidates_by_key = {}


def _get_gemini_client(api_key):
    if api_key not in _gemini_clients:
        _gemini_clients[api_key] = genai.Client(api_key=api_key)
    return _gemini_clients[api_key]


def _gemini_model_adaylarini_getir(client):
    try:
        adaylar = []
        for m in client.models.list():
            ad = getattr(m, "name", "") or ""
            ad = ad.split("/")[-1]
            if not ad:
                continue
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


def _gemini_next_model(api_key):
    client = _get_gemini_client(api_key)
    if api_key not in _gemini_candidates_by_key:
        _gemini_candidates_by_key[api_key] = _gemini_model_adaylarini_getir(client)
    if _gemini_model_by_key.get(api_key) is not None:
        return _gemini_model_by_key[api_key]
    adaylar = _gemini_candidates_by_key[api_key]
    if not adaylar:
        return None
    _gemini_model_by_key[api_key] = adaylar.pop(0)
    return _gemini_model_by_key[api_key]


def _gemini_model_basarisiz(api_key, hata):
    kod = getattr(hata, "code", None)
    metin = str(hata).lower()
    model_sorunu = kod == 404 or "404" in metin or "not found" in metin or "not supported" in metin
    if model_sorunu:
        _gemini_model_by_key[api_key] = None
        return True
    return False


def ai_yorum(baglam, api_key):
    """
    Verilen "baglam" metnini Gemini'ye gonderip 2-3 cumlelik teknik yorum
    ister. Donus: (yorum, uyari_mesaji) - ikisinden en fazla biri dolu olur.
    api_key bossa ya da paket kurulu degilse sessizce (None, None) doner -
    UI tarafinda bu durumda AI yorum bolumu hic gosterilmez.
    """
    if not _GEMINI_AVAILABLE:
        return None, "AI yorumu icin 'pip install google-genai' gerekiyor."
    if not api_key:
        return None, None

    prompt = (
        "Sen tecrubeli bir futbol scoutu ve teknik analistsin. Asagida "
        "bir veri analizi sorgusunun ciktisi var. Buna dayanarak KISA "
        "(2-3 cumle), somut ve teknik bir yorum yaz. Genel gecer, "
        "herkese uyan cumleler kurma - verideki gercek sayilara/"
        "isimlere/pozisyonlara atif yap. Guclu yanlarin yani sira "
        "eksik/zayif gorunen bir noktaya da deginmekten cekinme. "
        "Turkce yaz.\n\n---\n" + baglam
    )
    for _ in range(4):
        model_adi = _gemini_next_model(api_key)
        if model_adi is None:
            return None, "Uygun bir Gemini modeli bulunamadi."
        try:
            client = _get_gemini_client(api_key)
            response = client.models.generate_content(model=model_adi, contents=prompt)
            return response.text.strip(), None
        except Exception as e:
            if _gemini_model_basarisiz(api_key, e):
                continue
            return None, f"AI yorumu alinamadi: {e}"
    return None, "Denenen Gemini modellerinin hicbiri calismadi."


def ai_yorum_goster(baglam, key_prefix):
    """Sonuc tablosunun altina AI yorum bolumunu (varsa) ekler. Hata olursa
    sessiz bir bilgi notu gosterir, ana sonucu ASLA etkilemez."""
    api_key = st.session_state.get("gemini_api_key", "").strip()
    if not api_key:
        return
    with st.spinner("AI teknik yorum aliniyor..."):
        yorum, hata = ai_yorum(baglam, api_key)
    if yorum:
        st.markdown("**🤖 AI Teknik Yorum**")
        st.info(yorum)
    elif hata:
        st.caption(f"(AI yorum atlandi: {hata})")


# ==========================================================================
# VERI YUKLEME (cache'li - ayni dosya birden fazla kez okunmasin)
# ==========================================================================

@st.cache_data(show_spinner="Oyuncu verisi yukleniyor...")
def load_player_norm():
    return pd.read_csv(os.path.join(DATA_DIR, "player_season_normalized.csv"))


@st.cache_data(show_spinner="Takim verisi yukleniyor...")
def load_team_norm():
    return pd.read_csv(os.path.join(DATA_DIR, "team_season_tactics_normalized.csv"))


@st.cache_data(show_spinner="Takim isimleri yukleniyor...")
def load_team_names():
    df = pd.read_csv(os.path.join(DATA_DIR, "teamss.csv"))
    df = df.rename(columns={"wyId": "teamId"})
    return df[["teamId", "team_name"]]


_SLIM_EVENTS_COLS = [
    "matchId", "teamId", "playerId", "eventName", "subEventName",
    "tag_1801", "pos_orig_x", "pos_orig_y", "pos_dest_x", "pos_dest_y",
    "matchPeriod", "eventSec", "match_dateutc",
]
_SLIM_EVENTS_DTYPES = {
    "matchId": "int32", "teamId": "int32", "playerId": "int32",
    "tag_1801": "float32",
    "pos_orig_x": "float32", "pos_orig_y": "float32",
    "pos_dest_x": "float32", "pos_dest_y": "float32",
    "eventSec": "float32",
}


def _ensure_slim_events_file(slim_path, full_path):
    """
    Pas Agi'nin ihtiyac duydugu ~13 sutunu, buyuk ham event dosyasindan
    (eng_players_data_prep.csv, ~1.4 GB) cikarip cok daha kucuk,
    sikistirilmis bir dosyaya (event_pass_network_slim.csv.gz) yazar -
    TUM SATIRLAR korunur, sadece gereksiz sutunlar atilir (Pas Agi hicbir
    veri kaybi olmadan calismaya devam eder). Boylece bu kucuk dosya
    GitHub/Streamlit Cloud'a da sigacak kadar kucuk olur.

    Ayri bir script calistirmaya GEREK YOK - bu fonksiyon, "Pas Agi"
    sekmesine ilk kez tiklandiginda otomatik calisir; dosya bir kere
    uretildikten sonra bir daha calismaz (dosya zaten var, direkt okunur).
    Buyuk dosyayi PARCA PARCA (chunksize) okur, RAM sorunu yasanmaz.
    """
    if os.path.exists(slim_path) or not os.path.exists(full_path):
        return

    header_cols = pd.read_csv(full_path, nrows=0).columns.tolist()
    name_col = "match_match_name" if "match_match_name" in header_cols else (
        "match_name" if "match_name" in header_cols else None)
    usecols = _SLIM_EVENTS_COLS + ([name_col] if name_col else [])

    tmp_path = os.path.join(DATA_DIR, "_event_pass_network_slim_tmp.csv")
    first = True
    for chunk in pd.read_csv(full_path, usecols=usecols, dtype=_SLIM_EVENTS_DTYPES,
                              chunksize=500_000, low_memory=False):
        chunk.to_csv(tmp_path, mode=("w" if first else "a"), header=first, index=False)
        first = False

    with open(tmp_path, "rb") as f_in, gzip.open(slim_path, "wb", compresslevel=6) as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(tmp_path)


@st.cache_data(show_spinner="Event verisi hazirlaniyor (ilk seferde birkac dakika surebilir, sonrasinda hizli olur)...")
def load_events():
    """
    Pas Agi icin gereken event verisini okur. Once KUCUK/hizli ozel dosyayi
    (event_pass_network_slim.csv.gz) arar; yoksa ve buyuk ham dosya
    (eng_players_data_prep.csv) varsa, onu OTOMATIK olarak kucultup
    diske yazar (bir sonraki calistirmada, ve deploy edilen surumde de
    dogrudan bu kucuk dosya kullanilir) - ayri bir script calistirmaya
    gerek yok.
    """
    slim_path = os.path.join(DATA_DIR, "event_pass_network_slim.csv.gz")
    full_path = os.path.join(DATA_DIR, "eng_players_data_prep.csv")

    _ensure_slim_events_file(slim_path, full_path)

    if os.path.exists(slim_path):
        df = pd.read_csv(slim_path, compression="gzip", low_memory=False)
    else:
        df = pd.read_csv(full_path, low_memory=False)

    for col in ["matchId", "teamId", "playerId"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ==========================================================================
# ISIMDEN ID / KAYIT BULMA — query_runner.py'deki resolve_* fonksiyonlarinin
# AYNI arama mantigi, ama Streamlit'e uygun olarak print() yerine eslesen
# adaylarin LISTESINI donduruyor; disambiguation UI tarafinda yapiliyor.
# ==========================================================================

TEAM_NICKNAMES = {
    "barca": "barcelona", "atleti": "atletico madrid", "depor": "deportivo",
    "juve": "juventus", "inter": "internazionale", "psg": "paris saint germain",
    "om": "marseille", "ol": "lyon", "bvb": "borussia dortmund",
    "man utd": "manchester united", "man united": "manchester united",
    "man city": "manchester city", "spurs": "tottenham",
}


def find_team_matches(query, team_names_df, valid_ids=None):
    """Takim adindan aday listesi (teamId, team_name) dondurur."""
    df = team_names_df.dropna(subset=["team_name"]).copy()
    if valid_ids is not None:
        df = df[df["teamId"].isin(valid_ids)]
    df = df.drop_duplicates(subset=["teamId"])

    query_lower = query.strip().lower()
    if not query_lower:
        return []

    exact = df[df["team_name"].str.lower() == query_lower]
    if len(exact) >= 1:
        row = exact.iloc[0]
        return [(row["teamId"], row["team_name"])]

    def word_search(text):
        words = [w for w in text.lower().split() if w]
        def all_words_match(name):
            return all(w in str(name).lower() for w in words)
        return df[df["team_name"].apply(all_words_match)].drop_duplicates(subset=["team_name"])

    aday = word_search(query_lower)
    if len(aday) == 0 and query_lower in TEAM_NICKNAMES:
        aday = word_search(TEAM_NICKNAMES[query_lower])

    return list(zip(aday["teamId"].tolist(), aday["team_name"].tolist()))


def find_player_matches(query, norm):
    """Oyuncu adindan aday (tam player_name) listesi dondurur."""
    isimler = norm["player_name"].dropna().drop_duplicates()
    query_lower = query.strip().lower()
    if not query_lower:
        return []

    exact = isimler[isimler.str.lower() == query_lower]
    if len(exact) >= 1:
        return [exact.iloc[0]]

    query_words = [w for w in query_lower.split() if w]

    def all_words_match(name):
        return all(w in str(name).lower() for w in query_words)

    return isimler[isimler.apply(all_words_match)].tolist()[:30]


def pick_from_matches(matches, label, key, display=lambda m: m):
    """
    Birden fazla aday varsa bir selectbox gosterip secileni dondurur; tek
    aday varsa dogrudan onu dondurur; hic aday yoksa None (ve bir uyari).
    """
    if len(matches) == 0:
        st.warning(f"'{label}' icin eslesme bulunamadi. 'Lige Gore Listele' sekmesinden gecerli isimleri kontrol edebilirsin.")
        return None
    if len(matches) == 1:
        return matches[0]
    st.info(f"Tam eslesme yok, {len(matches)} aday bulundu - birini sec:")
    secim = st.selectbox("Adaylar", options=matches, format_func=display, key=key)
    return secim


# ==========================================================================
# MOTORLAR — query_runner.py'deki fonksiyonlarla AYNI hesap mantigi;
# sadece disambiguation'i disariya (yukaridaki find_*_matches) tasidik ve
# matplotlib figurlerini kaydetmek yerine dondurduk (st.pyplot icin).
# ==========================================================================

def scout_similar_players(norm, player_name, top_n=10):
    """SCOUT MOTORU: ayni pozisyon grubunda (600+ dakika havuzu icinde)
    oyun stiline gore kumeler, cosine similarity ile en benzer top_n
    oyuncuyu dondurur. player_name'in TAM (zaten cozulmus) isim oldugu
    varsayilir."""
    feature_cols = [c for c in norm.columns if c.endswith("_p90_z")]
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

    matched = norm[norm["player_name"] == player_name]
    if matched.empty:
        return None
    target = matched.iloc[0]
    role = target["role_code"]
    target_vector = X_all.loc[[target.name]]

    candidate_pool = pool[pool["role_code"] == role]
    candidate_vectors = pool_X.loc[candidate_pool.index]
    similarities = cosine_similarity(target_vector, candidate_vectors)[0]

    result_cols = ["player_name", "role_code", "style_cluster", "minutes_played"]
    if "league_code" in candidate_pool.columns:
        result_cols.insert(2, "league_code")
    result = candidate_pool[result_cols].copy()
    result["similarity"] = similarities
    result = result[result["player_name"] != player_name]
    return result.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)


def tactic_similar_teams(norm, team_id, top_n=5):
    """TAKTIK MOTORU: takimlari oyun tarzina gore kumeler, en benzer
    top_n takimi dondurur."""
    feature_cols = [c for c in norm.columns if c.endswith("_z")]
    X = norm[feature_cols].fillna(0)

    n_clusters = 3
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    norm = norm.copy()
    norm["style_cluster"] = km.fit_predict(X)

    idx = norm[norm["teamId"] == team_id].index[0]
    sims = cosine_similarity(X.loc[[idx]], X)[0]
    result_cols = ["teamId", "style_cluster"]
    if "league_code" in norm.columns:
        result_cols.insert(1, "league_code")
    result = norm[result_cols].copy()
    result["similarity"] = sims
    result = result[result["teamId"] != team_id]
    return result.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)


RADAR_METRICS = {
    "FW": {"goals_p90": "Gol", "shots_on_target_p90": "Isabetli Sut", "xt_added_total_p90": "xT",
           "key_passes_p90": "Kilit Pas", "take_ons_won_p90": "Basarili Calim",
           "assists_p90": "Asist", "aerial_duels_won_p90": "Hava Toplari"},
    "MD": {"passes_accurate_p90": "Isabetli Pas", "key_passes_p90": "Kilit Pas",
           "through_passes_p90": "Kesme Pas", "xt_added_total_p90": "xT",
           "interceptions_p90": "Top Kapma", "duels_won_p90": "Kazanilan Duello",
           "take_ons_won_p90": "Basarili Calim"},
    "DF": {"duels_won_p90": "Kazanilan Duello", "aerial_duels_won_p90": "Hava Toplari",
           "interceptions_p90": "Top Kapma", "clearances_p90": "Uzaklastirma",
           "sliding_tackles_p90": "Kayarak Mudahale", "passes_accurate_p90": "Isabetli Pas",
           "blocks_p90": "Blok"},
    "GK": {"reflex_saves_p90": "Refleks Kurtaris", "save_attempts_p90": "Kurtaris Denemesi",
           "passes_accurate_p90": "Isabetli Pas", "long_passes_p90": "Uzun Pas",
           "gk_leaving_line_p90": "Cizgi Disi Mudahale", "goals_conceded_event_p90": "Az Gol Yeme"},
}
RADAR_INVERT_METRICS = {"goals_conceded_event_p90"}


def _radar_percentiles(norm, player_name):
    row = norm[norm["player_name"] == player_name]
    if row.empty:
        return None, None
    row = row.iloc[0]
    role = row["role_code"]
    metrics = RADAR_METRICS[role]
    values = []
    for metric in metrics:
        pct = row.get(f"{metric}_percentile")
        if metric in RADAR_INVERT_METRICS and pd.notnull(pct):
            pct = 100 - pct
        values.append(pct if pd.notnull(pct) else 0)
    return metrics, values


def player_radar_fig(norm, player_names):
    """Radar grafigini matplotlib fig olarak dondurur (st.pyplot ile
    gosterilir), diske yazmaz."""
    metrics, _ = _radar_percentiles(norm, player_names[0])
    if metrics is None:
        return None
    labels = list(metrics.values())
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 7), subplot_kw=dict(polar=True))
    colors = ["firebrick", "steelblue", "seagreen"]
    for i, name in enumerate(player_names):
        m_i, values = _radar_percentiles(norm, name)
        if m_i is None:
            continue
        values = values + values[:1]
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=name)
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    fig.suptitle(" vs ".join(player_names) + "\npercentile radar (pozisyon grubu icinde)", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def player_full_stats_fig(norm, player_name, top_n=25):
    row = norm[norm["player_name"] == player_name]
    if row.empty:
        return None
    row = row.iloc[0]
    pct_cols = [c for c in norm.columns if c.endswith("_p90_percentile")]
    data = row[pct_cols].dropna().astype(float)
    data.index = [c.replace("_p90_percentile", "") for c in data.index]
    data = data.sort_values(ascending=True)
    if top_n:
        data = data.tail(top_n)

    fig, ax = plt.subplots(figsize=(7, max(5, len(data) * 0.22)))
    colors = plt.cm.RdYlGn(data.values / 100)
    ax.barh(data.index, data.values, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile (pozisyon grubu icinde)")
    ax.set_title(f"{player_name} - Tam Istatistik Ozeti")
    plt.tight_layout()
    return fig


def pass_network_fig(ev, match_id, team_id):
    """PAS AGI + bolge yogunlugu figurunu dondurur (diske yazmaz)."""
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
        key = tuple(sorted([row["playerId"], row["receiver_playerId"]]))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    pass_volume = passes["playerId"].value_counts().add(
        passes["receiver_playerId"].value_counts(), fill_value=0
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
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
    return fig, passes, pass_volume


def pas_agi_ozet_metni(ev, match_id, team_id, passes, pass_volume):
    """AI yorum icin sayisal ozet metni (pass_network_fig ile ayni
    hesaptan turetilir, tekrar hesaplamaz)."""
    team_ev = ev[(ev["matchId"] == match_id) & (ev["teamId"] == team_id)]
    is_pass = team_ev["eventName"] == "Pass"
    is_pass_ok = is_pass & (team_ev["tag_1801"] == 1)
    toplam_pas = int(is_pass.sum())
    isabetli_pas = int(is_pass_ok.sum())
    isabet_pct = (isabetli_pas / toplam_pas * 100) if toplam_pas > 0 else 0

    pair_counts = {}
    for _, row in passes.iterrows():
        key = tuple(sorted([int(row["playerId"]), int(row["receiver_playerId"])]))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    en_iyi_ikili = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    en_hareketli = pass_volume.sort_values(ascending=False).head(3)

    satirlar = [
        f"Toplam pas denemesi: {toplam_pas}, isabetli: {isabetli_pas} (%{isabet_pct:.1f} isabet)",
        "En cok pas alisverisi yapan ikili(ler): "
        + ", ".join(f"({a}-{b}: {c})" for (a, b), c in en_iyi_ikili),
        "En hareketli oyuncular (playerId, toplam pas hacmi): "
        + ", ".join(f"{pid}: {int(v)}" for pid, v in en_hareketli.items()),
    ]
    return "\n".join(satirlar)


def fig_download_button(fig, filename, label="PNG indir"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="image/png")


# ==========================================================================
# ARAYUZ
# ==========================================================================

st.title("⚽ Futbol Analitik Dashboard")
st.caption("5 lig (Ingiltere/Ispanya/Italya/Almanya/Fransa) - Scout Motoru, Taktik Motoru, Oyuncu Gelisimi, Pas Agi")

with st.sidebar:
    st.header("Ayarlar")
    if "gemini_api_key" not in st.session_state:
        st.session_state["gemini_api_key"] = ""
    st.session_state["gemini_api_key"] = st.text_input(
        "Google Gemini API Key (opsiyonel)",
        value=st.session_state["gemini_api_key"],
        type="password",
        help="Bos birakabilirsin - o zaman sorgu sonuclari AI yorum olmadan gosterilir. "
             "Ucretsiz key: https://aistudio.google.com/app/apikey",
    )
    if not _GEMINI_AVAILABLE:
        st.caption("⚠️ 'google-genai' paketi kurulu degil, AI yorum kapali. (pip install google-genai)")
    st.divider()
    st.caption(f"Veri klasoru: `{DATA_DIR}`")

tab_scout, tab_dev, tab_tactic, tab_pass, tab_browse = st.tabs(
    ["🔍 Scout Motoru", "📊 Oyuncu Gelisimi", "♟️ Taktik Motoru", "🕸️ Pas Agi", "📋 Lige Gore Listele"]
)

# --------------------------------------------------------------------
# TAB 1: SCOUT MOTORU
# --------------------------------------------------------------------
with tab_scout:
    st.subheader("Scout Motoru — benzer oyuncu bul")
    player_norm = load_player_norm()

    col1, col2 = st.columns([3, 1])
    isim_query = col1.text_input("Oyuncu adi", placeholder="ornek: Eden Hazard", key="scout_isim")
    head = col2.number_input("Kac benzer oyuncu?", min_value=1, max_value=50, value=10, key="scout_head")

    if st.button("Ara", key="scout_ara") and isim_query:
        matches = find_player_matches(isim_query, player_norm)
        secilen = pick_from_matches(matches, isim_query, key="scout_pick")
        if secilen:
            sonuc = scout_similar_players(player_norm, secilen, top_n=int(head))
            if sonuc is None or sonuc.empty:
                st.warning("Sonuc bulunamadi.")
            else:
                st.success(f"'{secilen}' oyuncusuna en benzer {len(sonuc)} oyuncu:")
                st.dataframe(sonuc, use_container_width=True)
                st.download_button("CSV indir", sonuc.to_csv(index=False).encode("utf-8"),
                                    file_name=f"scout_{secilen.replace(' ', '_')}.csv")
                baglam = (
                    f"Sorgulanan oyuncu: {secilen}\n"
                    f"Ayni pozisyondan en benzer {len(sonuc)} oyuncu (isim, pozisyon, "
                    f"[lig,] stil kumesi, dakika, benzerlik skoru):\n{sonuc.to_string(index=False)}"
                )
                ai_yorum_goster(baglam, key_prefix="scout")

# --------------------------------------------------------------------
# TAB 2: OYUNCU GELISIMI (RADAR)
# --------------------------------------------------------------------
with tab_dev:
    st.subheader("Oyuncu Gelisimi — radar grafik + tam istatistik ozeti")
    player_norm = load_player_norm()

    isimler_raw = st.text_input(
        "Oyuncu adi(lari) — karsilastirma icin virgulle ayir (en fazla 3)",
        placeholder="ornek: Eden Hazard, Mohamed Salah", key="dev_isimler",
    )

    if st.button("Goster", key="dev_goster") and isimler_raw:
        isim_listesi = [i.strip() for i in isimler_raw.split(",") if i.strip()][:3]
        cozulen_isimler = []
        for isim in isim_listesi:
            matches = find_player_matches(isim, player_norm)
            secilen = pick_from_matches(matches, isim, key=f"dev_pick_{isim}")
            if secilen:
                cozulen_isimler.append(secilen)

        if cozulen_isimler:
            fig = player_radar_fig(player_norm, cozulen_isimler)
            if fig:
                st.pyplot(fig)
                fig_download_button(fig, f"radar_{'_'.join(cozulen_isimler).replace(' ', '_')}.png")

            if len(cozulen_isimler) == 1:
                fig2 = player_full_stats_fig(player_norm, cozulen_isimler[0])
                if fig2:
                    st.pyplot(fig2)
                    fig_download_button(fig2, f"tam_istatistik_{cozulen_isimler[0].replace(' ', '_')}.png")

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
                    "100 = ayni pozisyondaki oyuncular arasinda en iyi):\n" + "\n".join(satirlar)
                )
                ai_yorum_goster(baglam, key_prefix="dev")

# --------------------------------------------------------------------
# TAB 3: TAKTIK MOTORU
# --------------------------------------------------------------------
with tab_tactic:
    st.subheader("Taktik Motoru — benzer takim bul")
    team_norm = load_team_norm()
    team_names = load_team_names()

    col1, col2 = st.columns([3, 1])
    takim_query = col1.text_input("Takim adi", placeholder="ornek: Chelsea", key="tactic_isim")
    top_n = col2.number_input("Kac benzer takim?", min_value=1, max_value=20, value=5, key="tactic_topn")

    if st.button("Ara", key="tactic_ara") and takim_query:
        matches = find_team_matches(takim_query, team_names, valid_ids=team_norm["teamId"].unique())
        secilen = pick_from_matches(matches, takim_query, key="tactic_pick", display=lambda m: m[1])
        if secilen:
            team_id, bulunan_isim = secilen
            sonuc = tactic_similar_teams(team_norm, team_id, top_n=int(top_n))
            sonuc = sonuc.merge(team_names, on="teamId", how="left")
            cols = ["teamId", "team_name"] + [c for c in sonuc.columns if c not in ("teamId", "team_name")]
            st.success(f"'{bulunan_isim}' takimina en benzer {len(sonuc)} takim:")
            st.dataframe(sonuc[cols], use_container_width=True)
            st.download_button("CSV indir", sonuc[cols].to_csv(index=False).encode("utf-8"),
                                file_name=f"taktik_{bulunan_isim.replace(' ', '_')}.csv")
            baglam = (
                f"Sorgulanan takim: {bulunan_isim}\n"
                f"Taktik/stil acisindan en benzer {len(sonuc)} takim (isim, [lig,] "
                f"stil kumesi, benzerlik skoru):\n{sonuc[cols].to_string(index=False)}"
            )
            ai_yorum_goster(baglam, key_prefix="tactic")

# --------------------------------------------------------------------
# TAB 4: PAS AGI
# --------------------------------------------------------------------
with tab_pass:
    st.subheader("Pas Agi — belirli mac + takim icin pas agi haritasi")
    st.caption("Bu sekme buyuk event dosyasini (eng_players_data_prep.csv) okur - ilk acilista birkac dakika surebilir.")

    takim_query = st.text_input("Takim adi", placeholder="ornek: Chelsea", key="pass_isim")

    if st.button("Maclari Getir", key="pass_getir") and takim_query:
        events = load_events()
        team_names = load_team_names()
        matches = find_team_matches(takim_query, team_names, valid_ids=events["teamId"].dropna().unique())
        secilen = pick_from_matches(matches, takim_query, key="pass_pick", display=lambda m: m[1])
        if secilen:
            team_id, bulunan_isim = secilen
            st.session_state["pass_team_id"] = team_id
            st.session_state["pass_team_name"] = bulunan_isim

            takim_ev = events[events["teamId"] == team_id]
            name_col = "match_match_name" if "match_match_name" in takim_ev.columns else "match_name"
            mac_cols = [c for c in ["matchId", name_col, "match_dateutc"] if c in takim_ev.columns]
            maclar = takim_ev[mac_cols].drop_duplicates(subset=["matchId"])
            if "match_dateutc" in maclar.columns:
                maclar = maclar.sort_values("match_dateutc")
            maclar = maclar.reset_index(drop=True)
            st.session_state["pass_maclar"] = maclar
            st.session_state["pass_name_col"] = name_col

    if "pass_maclar" in st.session_state and not st.session_state["pass_maclar"].empty:
        maclar = st.session_state["pass_maclar"]
        name_col = st.session_state["pass_name_col"]
        bulunan_isim = st.session_state["pass_team_name"]
        team_id = st.session_state["pass_team_id"]

        def mac_label(i):
            row = maclar.loc[i]
            tarih = row["match_dateutc"] if "match_dateutc" in maclar.columns else ""
            ad = row[name_col] if name_col in maclar.columns else f"matchId={row['matchId']}"
            return f"{tarih}  {ad}"

        secim_idx = st.selectbox(
            f"'{bulunan_isim}' takiminin maclari", options=maclar.index.tolist(),
            format_func=mac_label, key="pass_mac_secim",
        )

        if st.button("Pas Agini Ciz", key="pass_ciz"):
            match_id = int(maclar.loc[secim_idx, "matchId"])
            events = load_events()
            n_found = ((events["matchId"] == match_id) & (events["teamId"] == team_id)).sum()
            if n_found == 0:
                st.warning("Bu mac icin hicbir event bulunamadi.")
            else:
                fig, passes, pass_volume = pass_network_fig(events, match_id=match_id, team_id=team_id)
                st.pyplot(fig)
                fig_download_button(fig, f"pas_agi_{bulunan_isim.replace(' ', '_')}_{match_id}.png")
                ozet = pas_agi_ozet_metni(events, match_id, team_id, passes, pass_volume)
                st.text(ozet)
                baglam = f"Takim: {bulunan_isim}, matchId={match_id}\nPas agi ozet istatistikleri:\n{ozet}"
                ai_yorum_goster(baglam, key_prefix="pass")

# --------------------------------------------------------------------
# TAB 5: LIGE GORE LISTELE
# --------------------------------------------------------------------
with tab_browse:
    st.subheader("Lige Gore Listele — dogru oyuncu/takim adini bulmak icin")
    player_norm = load_player_norm()

    if "league_code" not in player_norm.columns:
        st.info("Bu veri setinde league_code kolonu yok (tek ligli calisilmis).")
    else:
        ligler = sorted(player_norm["league_code"].dropna().unique().tolist())
        lig = st.selectbox("Lig", options=ligler, key="browse_lig")
        secim = st.radio("Ne listelensin?", options=["Oyuncu", "Takim"], horizontal=True, key="browse_secim")

        if secim == "Takim":
            team_norm = load_team_norm()
            team_names = load_team_names()
            ornekler = team_norm.loc[team_norm["league_code"] == lig, ["teamId"]].drop_duplicates()
            ornekler = ornekler.merge(team_names, on="teamId", how="left")
            st.write(f"{lig} ligindeki takimlar ({len(ornekler)} takim):")
            st.dataframe(ornekler[["team_name", "teamId"]].sort_values("team_name"), use_container_width=True)
        else:
            ornekler = player_norm.loc[
                player_norm["league_code"] == lig, ["player_name", "role_code"]
            ].drop_duplicates().sort_values("player_name")
            st.write(f"{lig} ligindeki oyuncular ({len(ornekler)} oyuncu):")
            st.dataframe(ornekler, use_container_width=True)
