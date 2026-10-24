"""Step 1: define the prediction problem and build the model table."""
import numpy as np
import pandas as pd

# Step D1: Load the shipments and define the target
print("### D1")
df = pd.read_csv("data/shipments.csv",
                 parse_dates=["order_time", "dispatch_time", "delivery_time"])
df = df.sort_values("order_time").reset_index(drop=True)
first, last = df["order_time"].min(), df["order_time"].max()
print("Shipments:", len(df), "from", first.date(), "to", last.date())
print("Target: delivery_hours (hours from order to delivery)")
print(df["delivery_hours"].describe().round(1))

# Step D2: Build features that are known when the order arrives
print("### D2")
df["order_hour"] = df["order_time"].dt.hour
df["weekday"] = df["order_time"].dt.dayofweek
df["month"] = df["order_time"].dt.month
df["festive"] = df["month"].isin([10, 11]).astype(int)

def orders_last_24h(times):
    """Count earlier orders at the same hub in the previous 24 hours."""
    secs = (times - times.iloc[0]).dt.total_seconds().values
    start = np.searchsorted(secs, secs - 24 * 3600, side="left")
    return np.arange(len(secs)) - start

df["hub_load_24h"] = df.groupby("hub")["order_time"].transform(orders_last_24h)
features = ["hub", "zone", "shipping_mode", "product_category", "vehicle_type",
            "distance_km", "weight_kg", "parcels", "promised_hours",
            "order_hour", "weekday", "festive", "hub_load_24h"]
print("Features:", len(features))
print(df[features].head(3).T)

# Step D3: Split by time so the test set is the future
print("### D3")
cut = pd.Timestamp("2025-10-01")
df["split"] = np.where(df["order_time"] < cut, "train", "test")
for name, part in df.groupby("split"):
    print(name, len(part), "shipments, festive share:",
          f"{part['festive'].mean():.0%}", "late rate:",
          f"{part['late'].mean():.1%}")
df.to_csv("data/model_table.csv", index=False)
