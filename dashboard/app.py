import streamlit as st
import pandas as pd, plotly.graph_objects as go, joblib

st.set_page_config(page_title="NBA Shot Quality", page_icon="🏀", layout="wide")

st.title("🏀 NBA Shot Quality Predictor")
st.markdown("Predict how likely an NBA shot is to go in — and see where similar shots have been taken historically.")

# Load data and model once, cache so Streamlit doesn't reload on every slider move
@st.cache_data
def load_data():
  return pd.read_parquet("data/processed/shots_clean.parquet")

@st.cache_resource
def load_model():
  model = joblib.load("models/shot_model.pkl")
  feature_cols = joblib.load("models/feature_cols.pkl")
  return model, feature_cols

df = load_data()
model, feature_cols = load_model()

# --- Court drawing helper ---
def draw_court(fig):
  court_lines = [
    dict(type="circle", x0=-0.75, y0=-0.75, x1=0.75, y1=0.75, line=dict(color="white", width=2)),
    dict(type="rect", x0=-8, y0=-4.75, x1=8, y1=14.25, line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"),
    dict(type="circle", x0=-6, y0=7.75, x1=6, y1=19.75, line=dict(color="white", width=2)),
    dict(type="path", path="M -22 -4.75 L -22 9.25 Q 0 35 22 9.25 L 22 -4.75",
      line=dict(color="white", width=2)),
  ]
  for shape in court_lines:
    fig.add_shape(**shape)
  return fig

# --- Sliders (ABOVE the heatmap so it reacts to them) ---
st.header("Select a shot scenario")
col1, col2, col3 = st.columns(3)

with col1:
  dist = st.slider("Shot distance (ft)", 0, 35, 15,
    help="0 ft = at the rim, 24 ft ≈ three-point line")

with col2:
  period = st.selectbox("Game period", options=[1,2,3,4],
    format_func=lambda x: f"Quarter {x}",
    help="Which quarter of the game")

with col3:
  clock = st.slider("Seconds left on shot clock", 0, 24, 12,
    help="24 = fresh shot clock, 0 = buzzer beater")

# --- Model prediction ---
row = {col: 0 for col in feature_cols}
row.update({"SHOT_DISTANCE": dist, "PERIOD": period, "CLOCK_SECONDS": clock})
prob = model.predict_proba(pd.DataFrame([row]))[0][1]

st.divider()
res_col1, res_col2 = st.columns([1, 2])
with res_col1:
  st.metric("Expected FG%", f"{prob*100:.1f}%",
    help="Predicted probability this shot goes in")
with res_col2:
  if prob >= 0.55:
    st.success("🟢 High-quality look — close range or wide open")
  elif prob >= 0.40:
    st.warning("🟡 Average shot — typical mid-range or pull-up")
  else:
    st.error("🔴 Tough shot — contested, deep, or off-balance")

st.divider()

# --- Map distance slider to NBA shot zone ---
# NBA zones match how analysts actually categorize shots
def dist_to_zone(d):
  if d <= 4: return "Restricted Area"
  elif d <= 8: return "In The Paint (Non-RA)"
  elif d <= 14: return "Mid-Range"
  elif d <= 22: return "Mid-Range"
  else: return "Above the Break 3"

zone = dist_to_zone(dist)

# SHOT_ZONE_BASIC was one-hot encoded — find the matching column
zone_col = f"SHOT_ZONE_BASIC_{zone}"

st.header("Where are similar shots taken from?")

if zone_col not in df.columns:
  st.warning(f"Zone column '{zone_col}' not found — check your SHOT_ZONE_BASIC values with df['SHOT_ZONE_BASIC'].unique() before encoding.")
else:
  filtered = df[df[zone_col] == 1]
  sample = filtered.sample(min(len(filtered), 5000))
  made = sample[sample["SHOT_MADE_FLAG"] == 1]
  missed = sample[sample["SHOT_MADE_FLAG"] == 0]
  fg_pct = len(made) / len(sample) * 100
  st.caption(f"Zone: {zone} · {len(sample):,} shots shown · Historical FG%: {fg_pct:.1f}% · Green = made, Red = missed")

  fig = go.Figure()

  fig.add_trace(go.Scattergl(
    x=made["LOC_X"], y=made["LOC_Y"], mode="markers",
    marker=dict(color="rgba(0,200,0,0.35)", size=4),
    name="Made",
    hovertemplate="Made shot<extra></extra>"
  ))

  fig.add_trace(go.Scattergl(
    x=missed["LOC_X"], y=missed["LOC_Y"], mode="markers",
    marker=dict(color="rgba(220,0,0,0.25)", size=4),
    name="Missed",
    hovertemplate="Missed shot<extra></extra>"
  ))

  draw_court(fig)

  fig.update_layout(
    xaxis=dict(title="← Left Baseline · Court Width · Right Baseline →", range=[-30, 30]),
    yaxis=dict(title="Distance from Basket", range=[-5, 43]),
    plot_bgcolor="#1a3a1a",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.05),
    height=550,
  )
  st.plotly_chart(fig, use_container_width=True)
