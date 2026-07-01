import pandas as pd, mlflow, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

df = pd.read_parquet("data/processed/shots_clean.parquet")

# Drop raw time columns — CLOCK_SECONDS already captures this
df = df.drop(columns=["MINUTES_REMAINING", "SECONDS_REMAINING"], errors="ignore")

X = df.drop("SHOT_MADE_FLAG", axis=1)
y = df["SHOT_MADE_FLAG"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
  base = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, eval_metric="logloss")

  # Calibration wraps the model so predict_proba returns realistic probabilities
  # Without this, XGBoost tends to push predictions toward 0 or 1 too aggressively
  model = CalibratedClassifierCV(base, method="isotonic", cv=3)
  model.fit(X_train, y_train)

  preds = model.predict_proba(X_test)[:,1]
  auc = roc_auc_score(y_test, preds)
  loss = log_loss(y_test, preds)
  mean_pred = preds.mean()

  mlflow.log_metrics({"auc": auc, "log_loss": loss, "mean_pred_prob": mean_pred})
  print(f"AUC: {auc:.4f} | Log Loss: {loss:.4f} | Mean predicted FG%: {mean_pred*100:.1f}%")
  print("Mean predicted FG% should be close to actual FG% (~45-47%)")

  joblib.dump(model, "models/shot_model.pkl")
  joblib.dump(list(X.columns), "models/feature_cols.pkl")