import pandas as pd, mlflow, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

df = pd.read_parquet("data/processed/shots_clean.parquet")
X = df.drop("SHOT_MADE_FLAG", axis=1)
y = df["SHOT_MADE_FLAG"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
  model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    eval_metric="auc",
    early_stopping_rounds=20,
    random_state=42
)

  model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
  )
  preds = model.predict_proba(X_test)[:,1]
  auc = roc_auc_score(y_test, preds)
  loss = log_loss(y_test, preds)
  mlflow.log_metrics({"auc": auc, "log_loss": loss})
  print(f"AUC: {auc:.4f} | Log Loss: {loss:.4f}")
  joblib.dump(model, "models/shot_model.pkl")
  joblib.dump(list(X.columns), "models/feature_cols.pkl")