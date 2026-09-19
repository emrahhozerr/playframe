# PlayFrame

Player and team similarity analysis from Wyscout match-event data: **Scout Engine**, **Tactical Engine**,
**Player Development**.

Miuul Data Science Bootcamp capstone project - Team: **DataGambit (G04)** - Mehmet Can Basaran, Yaser Girit, Ibrahim Emrah Ozer

## What this project does

PlayFrame processes raw match-event data from Europe's top 5 leagues (England, Spain, Italy, Germany,
France), 2017/18 season, end-to-end into three analysis engines:

- **Scout Engine** - for a given player, finds the statistically most similar players within the same
  position group, across all 5 leagues.
- **Tactical Engine** - for a given team, finds the most similar teams in terms of playing style.
- **Player Development** - summarizes player performance visually with radar charts and pass-network
  visualizations.

Results are served through an interactive Streamlit dashboard.

## Dataset

Wyscout Soccer Match Event Dataset (Pappalardo, L., Cintia, P., Rossi, A., Massucco, E., Ferragina, P.,
Pedreschi, D., & Giannotti, F. (2019). *A public data set of spatio-temporal match events in soccer
competitions*. Scientific Data, 6, 236. https://www.nature.com/articles/s41597-019-0247-7).

Scope used in this project: 5 leagues (no international tournaments), 2017/18 season, 98 teams,
2,569 players.

The raw data is not included in this repo due to size constraints. It can be downloaded from this
Kaggle mirror: https://www.kaggle.com/datasets/aleespinosa/soccer-match-event-dataset

## Folder structure

| Folder | Contents |
|---|---|
| `1-Data Enginers` | Fetching/merging the raw CSVs |
| `2-EDA` | Exploratory data analysis |
| `3-Data Preprocessing` | Data cleaning (own-goal fix, missing records) |
| `4-Feature Engineer` | per90, xT, normalization, development of the Scout/Tactical engines |
| `5-unsupervised learning` | Clustering experiments |
| `6-Pipline` | End-to-end pipeline scripts + the **Streamlit dashboard** (the live app runs from here) |

## Methodology summary

- **per90**: Raw event counts (passes, shots, duels, fouls...) are scaled to 90 minutes played, so
  players/teams can be compared fairly regardless of minutes.
- **xT (Expected Threat)**: The pitch is split into a 16x12 grid; the added threat of each pass/dribble
  is computed iteratively via a transition matrix over pitch zones.
- **Normalization**: On the player side, z-scores are computed within each position group; on the team
  side, all teams are z-scored as a single group.
- **Similarity ranking**: The actual ranking is driven by **cosine similarity**. KMeans additionally
  produces a "playing style" cluster label - this is purely descriptive context and does not affect the
  ranking.

## Running the dashboard locally

```
cd 6-Pipline
pip install -r requirements.txt
streamlit run dashboard.py
```

You can optionally enter a Google Gemini API key in the sidebar to enable AI-generated commentary; if left
blank, that section is silently hidden and everything else works normally.

## Live demo

[https://playframe-mzzpqst9kzfntbai8fjwfb.streamlit.app/](https://playframe-mzzpqst9kzfntbai8fjwfb.streamlit.app/)

## License

MIT - see [LICENSE](LICENSE)

## References

Pappalardo, L., Cintia, P., Rossi, A., Massucco, E., Ferragina, P., Pedreschi, D., & Giannotti, F. (2019).
A public data set of spatio-temporal match events in soccer competitions. *Scientific Data*, 6, 236.
