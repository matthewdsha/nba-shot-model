from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams
import pandas as pd, time, os
from tenacity import retry, stop_after_attempt, wait_exponential

SEASONS = ["2019-20","2020-21","2021-22","2022-23","2023-24","2024-25"]
CACHE_DIR = "data/raw"

HEADERS = {
  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
  "Referer": "https://www.nba.com/",
  "Accept": "application/json, text/plain, */*",
  "Accept-Language": "en-US,en;q=0.9",
  "Origin": "https://www.nba.com",
  "Connection": "keep-alive",
}

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def fetch_season(season: str) -> pd.DataFrame:
  sc = shotchartdetail.ShotChartDetail(
    team_id=0, player_id=0,
    season_type_all_star="Regular Season",
    season_nullable=season,
    context_measure_simple="FGA",  # FGA = all attempts, FGM = made only (bad default)
    timeout=60,
    headers=HEADERS
  )
  return sc.get_data_frames()[0]

def fetch_with_cache(season: str) -> pd.DataFrame:
  cache_file = f"{CACHE_DIR}/shots_{season}.parquet"
  if os.path.exists(cache_file):
    print(f" {season} already cached, skipping.")
    return pd.read_parquet(cache_file)
  df = fetch_season(season)
  df.to_parquet(cache_file)
  return df

for s in SEASONS:
  print(f"Fetching {s}...")
  df = fetch_with_cache(s)
  print(f" Done — {len(df):,} shots")
  time.sleep(3) # be polite to the API