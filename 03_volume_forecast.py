"""Step 3: forecast daily shipment volume for each hub."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid")
results = {}

# Step V1: Build a daily volume table with calendar and lag features
print("### V1")
df = pd.read_csv("data/model_table.csv", parse_dates=["order_time"])
df["date"] = df["order_time"].dt.normalize()
daily = df.groupby(["hub", "date"]).size().rename("volume").reset_index()
full = pd.MultiIndex.from_product(
    [daily["hub"].unique(), pd.date_range(df["date"].min(), df["date"].max())],
    names=["hub", "date"])
daily = daily.set_index(["hub", "date"]).reindex(full, fill_value=0).reset_index()

grp = daily.groupby("hub")["volume"]
daily["lag_7"] = grp.shift(7)
daily["lag_14"] = grp.shift(14)
daily["mean_7_to_13"] = grp.transform(lambda s: s.shift(7).rolling(7).mean())
daily["weekday"] = daily["date"].dt.dayofweek
daily["festive"] = daily["date"].dt.month.isin([10, 11]).astype(int)
daily["trend"] = (daily["date"] - daily["date"].min()).dt.days
daily = daily.dropna().reset_index(drop=True)
daily = pd.get_dummies(daily, columns=["hub"], dtype=int, drop_first=False)
daily["hub"] = daily[[c for c in daily if c.startswith("hub_")]].idxmax(axis=1).str[4:]
print("Rows:", len(daily), "Hubs:", daily["hub"].nunique())
print(daily.groupby("hub")["volume"].mean().round(1))

# Step V2: Train on the first year and test on the next six months
print("### V2")
cut = pd.Timestamp("2025-10-01")
features = ["lag_7", "lag_14", "mean_7_to_13", "weekday", "festive", "trend"] + [
    c for c in daily if c.startswith("hub_") and c != "hub"]
train = daily[daily["date"] < cut]
test = daily[daily["date"] >= cut].copy()

models = {"Naive (same day last week)": None,
          "Linear Regression": LinearRegression(),
          "Linear Regression, log target": LinearRegression(),
          "Gradient Boosting": HistGradientBoostingRegressor(
              learning_rate=0.05, max_depth=4, max_iter=250, random_state=42)}
rows, preds = [], {}
for name, model in models.items():
    if model is None:
        p = test["lag_7"].values
    elif "log" in name:
        fit = model.fit(train[features], np.log1p(train["volume"]))
        p = np.expm1(fit.predict(test[features]))
    else:
        p = model.fit(train[features], train["volume"]).predict(test[features])
    p = np.clip(p, 0, None)
    preds[name] = p
    err = test["volume"].values - p
    rows.append({"model": name, "MAE": np.abs(err).mean(),
                 "RMSE": (err ** 2).mean() ** 0.5,
                 "WAPE %": np.abs(err).sum() / test["volume"].sum() * 100})
table = pd.DataFrame(rows).set_index("model").round(2)
print(table)
results["volume_table"] = table.to_dict("index")

# Step V3: Check the forecast in the festive peak and save it
print("### V3")
best_name = table["MAE"].idxmin()
print("Selected model:", best_name)
results["selected_volume_model"] = best_name
test["forecast"] = preds[best_name]
for label, mask in [("Festive months", test["festive"] == 1),
                    ("Normal months", test["festive"] == 0)]:
    part = test[mask]
    wape = np.abs(part["volume"] - part["forecast"]).sum() / part["volume"].sum()
    bias = (part["forecast"].sum() - part["volume"].sum()) / part["volume"].sum()
    print(label, "WAPE:", f"{wape:.1%}", "Bias:", f"{bias:+.1%}")
    results["bias_" + label.split()[0].lower()] = round(bias * 100, 1)
    results["wape_" + label.split()[0].lower()] = round(wape * 100, 1)
peak = test[(test["date"] >= "2025-11-03") & (test["date"] <= "2025-11-09")]
print("Forecast and actual shipments, week of 3 November 2025")
print(peak.groupby("hub")[["forecast", "volume"]].mean().round(1))
test[["hub", "date", "volume", "forecast"]].to_csv("data/volume_forecast.csv",
                                                    index=False)
results["peak_forecast"] = peak.groupby("hub")["forecast"].mean().round(1).to_dict()
results["peak_actual"] = peak.groupby("hub")["volume"].mean().round(1).to_dict()

# Step Z: Chart and saved results
tot = test.groupby("date")[["volume", "forecast"]].sum()
fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.plot(tot.index, tot["volume"], color="#1F3864", lw=1.2, label="Actual")
ax.plot(tot.index, tot["forecast"], color="#C0392B", lw=1.2, label="Forecast")
ax.set_ylabel("Shipments per day (all hubs)")
ax.set_title("Daily shipment volume: forecast against actual on the test months")
ax.legend()
fig.autofmt_xdate()
plt.tight_layout(); plt.savefig("figures/v01_volume_forecast.png", dpi=170); plt.close()
with open("outputs/volume_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
