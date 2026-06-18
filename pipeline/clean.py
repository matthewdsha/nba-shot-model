import pandas as pd, glob
import numpy as np

files = glob.glob("data/raw/shots_*.parquet")
df = pd.concat([pd.read_parquet(f) for f in files])

df = df[[
  "SHOT_MADE_FLAG", "SHOT_DISTANCE",
  "LOC_X", "LOC_Y", "SHOT_TYPE",
  "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA",
  "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING",
  "ACTION_TYPE"
]].copy()

df["CLOCK_SECONDS"] = df["MINUTES_REMAINING"]*60 + df["SECONDS_REMAINING"]
df["SHOT_DISTANCE_SQ"] = df["SHOT_DISTANCE"] ** 2
df["SHOT_DISTANCE_CUBE"] = df["SHOT_DISTANCE"] ** 3
df["SECONDS_LEFT"] = (df["MINUTES_REMAINING"] * 60) + df["SECONDS_REMAINING"]
df["SHOT_CLOCK_PRESSURE"] = df["SECONDS_LEFT"] < 5  # end of period desperation shots
df["SHOT_ANGLE"] = np.arctan2(df["LOC_X"], df["LOC_Y"])
df["SHOT_DIST_POLAR"] = np.sqrt(df["LOC_X"]**2 + df["LOC_Y"]**2)
df["IS_THREE"] = (df["SHOT_TYPE"] == "3PT Field Goal").astype(int)
df["THREE_X_DIST"] = df["IS_THREE"] * df["SHOT_DISTANCE"]
df["IS_CORNER_THREE"] = (
    (df["IS_THREE"] == 1) & (df["LOC_Y"] < 50)
).astype(int)
df["IS_FOURTH"] = (df["PERIOD"] == 4).astype(int)
df["IS_OT"] = (df["PERIOD"] > 4).astype(int)
df = pd.get_dummies(df, columns=["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_TYPE", "ACTION_TYPE"])
df.dropna(inplace=True)
df.to_parquet("data/processed/shots_clean.parquet")
print(f"Saved {len(df):,} shots")
