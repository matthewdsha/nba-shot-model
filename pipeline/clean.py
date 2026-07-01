import pandas as pd, glob

files = glob.glob("data/raw/shots_*.parquet")
df = pd.concat([pd.read_parquet(f) for f in files])

# Drop ACTION_TYPE — too many unique values, causes schema mismatches across seasons
df = df[[
  "SHOT_MADE_FLAG", "SHOT_DISTANCE",
  "LOC_X", "LOC_Y", "SHOT_TYPE",
  "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA",
  "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING"
]].copy()

df.dropna(inplace=True)

# NBA API returns coordinates in tenths of a foot
# Valid half-court range: LOC_X -250 to 250, LOC_Y -50 to 420
# Shots outside this are bad data — filter them out
df = df[(df["LOC_X"].between(-250, 250)) & (df["LOC_Y"].between(-50, 420))]

df["CLOCK_SECONDS"] = df["MINUTES_REMAINING"]*60 + df["SECONDS_REMAINING"]
df["IS_THREE"] = (df["SHOT_TYPE"] == "3PT Field Goal").astype(int)

# One-hot encode zone columns
df = pd.get_dummies(df, columns=["SHOT_ZONE_BASIC","SHOT_ZONE_AREA","SHOT_TYPE"])

# Verify 3-pointers are present
three_pct = df["IS_THREE"].mean() * 100
print(f"3PT share: {three_pct:.1f}% (expect ~33-38%)")

df.to_parquet("data/processed/shots_clean.parquet")
print(f"Saved {len(df):,} shots")