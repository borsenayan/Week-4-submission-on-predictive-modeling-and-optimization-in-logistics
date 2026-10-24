"""Step 2: predict delivery time with several models and pick the best."""
import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (classification_report, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 110)
results = {}

# Step M1: Prepare the training and test data
print("### M1")
df = pd.read_csv("data/model_table.csv", parse_dates=["order_time"])
num_cols = ["distance_km", "weight_kg", "parcels", "promised_hours",
            "order_hour", "weekday", "festive", "hub_load_24h"]
cat_cols = ["hub", "zone", "shipping_mode", "product_category", "vehicle_type"]
target = "delivery_hours"

train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
X_train, y_train = train[num_cols + cat_cols], train[target]
X_test, y_test = test[num_cols + cat_cols], test[target]

prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
print("Training rows:", len(train), "Test rows:", len(test))
print("Numeric features:", len(num_cols), "Categorical features:", len(cat_cols))

# Step M2: Compare models with time series cross validation
print("### M2")
def make(model):
    return Pipeline([("prep", prep), ("model", model)])

candidates = {
    "Baseline (mean)": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=20,
                                           random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=10,
                                           n_jobs=-1, random_state=42),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
}
cv = TimeSeriesSplit(n_splits=5)
scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error",
           "r2": "r2"}
rows = []
for name, model in candidates.items():
    scores = cross_validate(make(model), X_train, y_train, cv=cv, scoring=scoring)
    rows.append({"model": name, "RMSE": -scores["test_rmse"].mean(),
                 "MAE": -scores["test_mae"].mean(), "R2": scores["test_r2"].mean()})
cv_table = pd.DataFrame(rows).set_index("model").round(3)
print(cv_table.sort_values("RMSE"))
results["cv_table"] = cv_table.to_dict("index")

# Step M3: Tune the best model
print("### M3")
param_space = {
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
    "model__max_depth": [3, 4, 6, 8, None],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__min_samples_leaf": [10, 20, 40, 80],
    "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
    "model__max_iter": [150, 250, 400],
}
search = RandomizedSearchCV(
    make(HistGradientBoostingRegressor(random_state=42)), param_space,
    n_iter=15, cv=TimeSeriesSplit(n_splits=3),
    scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
print("Best settings:", best)
print("Cross validated RMSE after tuning:", round(-search.best_score_, 3))
results["best_params"] = {k: (None if v is None else float(v)) for k, v in best.items()}
results["tuned_cv_rmse"] = round(-search.best_score_, 3)

# Step M4: Final test on the untouched future months
print("### M4")
final = search.best_estimator_
baseline = make(DummyRegressor(strategy="mean")).fit(X_train, y_train)
pred = final.predict(X_test)
pred_base = baseline.predict(X_test)

def metrics(y, p):
    return {"RMSE": mean_squared_error(y, p) ** 0.5,
            "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p)}

report = pd.DataFrame({"Baseline": metrics(y_test, pred_base),
                       "Tuned model": metrics(y_test, pred)}).T.round(3)
print(report)
results["test_metrics"] = report.to_dict("index")

test = test.assign(pred=pred, error=pred - y_test)
by_group = pd.DataFrame({
    "MAE hub": test.groupby("hub")["error"].apply(lambda e: e.abs().mean()),
    "bias hub": test.groupby("hub")["error"].mean()}).round(2)
print(by_group)
period = test.groupby(test["festive"].map({0: "normal months", 1: "festive months"}))
print("MAE by period:", period["error"].apply(lambda e: e.abs().mean()).round(2).to_dict())
results["mae_by_hub"] = by_group["MAE hub"].to_dict()
results["mae_by_period"] = period["error"].apply(lambda e: e.abs().mean()).round(2).to_dict()
test[["shipment_id", "pred"]].to_csv("data/test_predictions.csv", index=False)

# Step M5: Turn the forecast into a late delivery warning
print("### M5")
flag = (test["pred"] > test["promised_hours"]).astype(int)
print(classification_report(test["late"], flag, target_names=["on time", "late"],
                            digits=3))
auc = roc_auc_score(test["late"], test["pred"] - test["promised_hours"])
print("ROC AUC of predicted minus promised hours:", round(auc, 3))
rep = classification_report(test["late"], flag, output_dict=True)
results["late_precision"] = round(rep["1"]["precision"], 3)
results["late_recall"] = round(rep["1"]["recall"], 3)
results["late_f1"] = round(rep["1"]["f1-score"], 3)
results["late_auc"] = round(auc, 3)
results["late_accuracy"] = round(rep["accuracy"], 3)

# Step M6: Which features matter most
print("### M6")
sample = test.sample(4000, random_state=1)
imp = permutation_importance(final, sample[num_cols + cat_cols], sample[target],
                             scoring="neg_mean_absolute_error", n_repeats=3,
                             random_state=1, n_jobs=-1)
importance = pd.Series(imp.importances_mean, index=num_cols + cat_cols)
importance = importance.sort_values(ascending=False).round(2)
print(importance.head(8))
results["importance"] = importance.to_dict()

# Step Z: Charts and saved results
cv_plot = cv_table.drop(index="Baseline (mean)").sort_values("RMSE")
ax = cv_plot[["RMSE", "MAE"]].plot(kind="barh", figsize=(7, 3.8),
                                   color=["#1F3864", "#7FA6D6"])
ax.set_xlabel("Error in hours (lower is better)")
ax.set_title("Cross validated error of each model")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout(); plt.savefig("figures/m01_model_comparison.png", dpi=170); plt.close()

plt.figure(figsize=(6, 5))
s = test.sample(3000, random_state=2)
plt.scatter(s[target], s["pred"], s=6, alpha=0.35, color="#2E75B6")
top = max(s[target].max(), s["pred"].max())
plt.plot([0, top], [0, top], color="#C0392B", lw=1.5, label="Perfect prediction")
plt.xlabel("Actual delivery time (hours)")
plt.ylabel("Predicted delivery time (hours)")
plt.title("Predicted against actual on the test months")
plt.legend()
plt.tight_layout(); plt.savefig("figures/m02_actual_vs_predicted.png", dpi=170); plt.close()

plt.figure(figsize=(7, 3.8))
importance.head(10).iloc[::-1].plot(kind="barh", color="#2E5597")
plt.xlabel("Increase in MAE when the feature is shuffled (hours)")
plt.title("Permutation importance of the top features")
plt.tight_layout(); plt.savefig("figures/m03_importance.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(7, 3.8))
test["abs_error"] = test["error"].abs()
sns.barplot(data=test, x="hub", y="abs_error", hue=test["festive"].map(
    {0: "Normal months", 1: "Festive months"}), errorbar=None,
    order=["Pune", "Nagpur", "Sambhajinagar", "Nashik"], ax=ax)
ax.set_ylabel("Mean absolute error (hours)")
ax.set_xlabel("")
ax.legend(title="")
ax.set_title("Prediction error by hub and season")
plt.tight_layout(); plt.savefig("figures/m04_error_by_hub.png", dpi=170); plt.close()

with open("outputs/model_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
