"""Create a hypothetical shipment dataset for RegionalKart.

RegionalKart is the made up ecommerce company from my Week 1 report. This
version covers 18 months (October 2024 to March 2026) so that the test period
contains a full festive season. Order volume also grows a little over time.
I build the patterns myself (for example a slow hub and a festive peak) so I
can check that the models find them. All numbers are simulated.
"""
import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(2026)
N = 38000

# Order times: more orders in the festive months and fewer on Sundays
days = pd.date_range("2024-10-01", periods=547, freq="D")
weight_day = np.ones(len(days))
weight_day[days.month == 10] = 1.7
weight_day[days.month == 11] = 1.4
weight_day[days.month == 12] = 1.1
weight_day[days.dayofweek == 6] *= 0.7
weight_day *= 1 + 0.2 * np.arange(len(days)) / 365      # growth over time
order_day = rng.choice(days, N, p=weight_day / weight_day.sum())
order_time = pd.to_datetime(order_day) + pd.to_timedelta(
    rng.integers(0, 24 * 60, N), unit="m")
month = order_time.month
festive = np.isin(month, [10, 11])
weekday = order_time.dayofweek

# Where and how the shipment goes
hubs = ["Pune", "Nashik", "Nagpur", "Sambhajinagar"]
hub = rng.choice(hubs, N, p=[0.35, 0.25, 0.22, 0.18])
zone = rng.choice(["Urban", "Semi urban", "Rural"], N, p=[0.5, 0.3, 0.2])
dist = np.where(zone == "Urban", rng.gamma(2, 7.5, N),
        np.where(zone == "Semi urban", rng.gamma(3, 13, N), rng.gamma(3, 30, N)))
distance = np.clip(dist + 1, 1, 220).round(1)

mode_probs = {"Urban": [0.55, 0.30, 0.15], "Semi urban": [0.70, 0.27, 0.03],
              "Rural": [0.85, 0.145, 0.005]}
mode = np.empty(N, dtype=object)
for z, p in mode_probs.items():
    idx = zone == z
    mode[idx] = rng.choice(["Standard", "Express", "Same Day"], idx.sum(), p=p)

category = rng.choice(["Fashion", "Electronics", "Home", "Essentials"], N,
                      p=[0.30, 0.20, 0.20, 0.30])

# Shipment volume (parcels) and weight
parcels = np.ceil(rng.lognormal(0.6, 0.9, N))
bulk = rng.random(N) < 0.04
parcels[bulk] = np.ceil(rng.lognormal(4.0, 0.5, bulk.sum()))
avg_weight = pd.Series(category).map(
    {"Fashion": 0.6, "Electronics": 1.8, "Home": 3.5, "Essentials": 1.2}).values
weight = (parcels * avg_weight * rng.lognormal(0, 0.2, N)).round(1)

vehicle = np.where((weight <= 10) & (distance <= 25), "Bike",
          np.where(weight >= 150, "Truck", "Van"))

# Promised delivery time in hours
promise_table = {("Urban", "Standard"): 32, ("Semi urban", "Standard"): 34,
                 ("Rural", "Standard"): 38, ("Urban", "Express"): 15,
                 ("Semi urban", "Express"): 17, ("Rural", "Express"): 22}
promised = np.array([8 if m == "Same Day" else promise_table[(z, m)]
                     for z, m in zip(zone, mode)])

# Processing time at the hub (order received until the parcel leaves)
hub_base = pd.Series(hub).map(
    {"Pune": 8, "Nagpur": 9, "Sambhajinagar": 10, "Nashik": 17}).values
mode_factor = pd.Series(mode).map(
    {"Standard": 1.0, "Express": 0.6, "Same Day": 0.35}).values
week_factor = np.where(weekday == 0, 1.35, np.where(weekday == 5, 1.1, 1.0))
peak_factor = np.where(festive,
    pd.Series(hub).map({"Pune": 1.3, "Nagpur": 1.3, "Sambhajinagar": 1.35,
                        "Nashik": 1.7}).values, 1.0)
processing = (hub_base * mode_factor * week_factor * peak_factor
              * rng.lognormal(0, 0.35, N)).round(1)

# Transit time (hub to customer)
speed = pd.Series(vehicle).map({"Bike": 18, "Van": 28, "Truck": 24}).values
zone_speed = pd.Series(zone).map(
    {"Urban": 0.75, "Semi urban": 1.0, "Rural": 0.85}).values
moving = distance / (speed * zone_speed) * rng.lognormal(0, 0.2, N)
wait = np.where(mode == "Standard", rng.uniform(2, 16, N),
       np.where(mode == "Express", rng.uniform(0, 4, N), 0.0))
stops = np.where(zone == "Rural", rng.uniform(0, 6, N), 0.0)
loading = np.where(vehicle == "Truck", 2.0, 0.0)
# Occasional exceptions such as traffic jams or vehicle trouble
exception = np.where(rng.random(N) < 0.10, rng.uniform(4, 20, N), 0.0)
transit = (moving + wait + stops + loading + exception).round(1)

delivery_hours = (processing + transit).round(1)
late = (delivery_hours > promised).astype(int)

# Transport cost in rupees
per_km = pd.Series(vehicle).map({"Bike": 3.5, "Van": 9.0, "Truck": 18.0}).values
fixed = pd.Series(vehicle).map({"Bike": 35, "Van": 85, "Truck": 240}).values
per_kg = np.where(vehicle == "Bike", 1.2, 0.7)
cost_mode = pd.Series(mode).map({"Standard": 1.0, "Express": 1.5, "Same Day": 2.2}).values
cost_zone = pd.Series(zone).map({"Urban": 1.0, "Semi urban": 1.05, "Rural": 1.25}).values
cost = ((fixed + per_km * distance + per_kg * weight) * cost_mode * cost_zone
        * np.where(festive, 1.1, 1.0) * rng.normal(1, 0.08, N)).round(2)

# Customer rating drops when the parcel is late
excess = np.clip(delivery_hours - promised, 0, None)
rating = np.clip(np.round(4.5 - 1.1 * late - 0.01 * excess
                          + rng.normal(0, 0.6, N)), 1, 5).astype(int)

df = pd.DataFrame({
    "shipment_id": np.arange(500000, 500000 + N),
    "order_time": order_time,
    "dispatch_time": order_time + pd.to_timedelta(processing, unit="h"),
    "delivery_time": order_time + pd.to_timedelta(delivery_hours, unit="h"),
    "hub": hub, "zone": zone, "distance_km": distance,
    "shipping_mode": mode, "product_category": category,
    "vehicle_type": vehicle, "parcels": parcels.astype(int), "weight_kg": weight,
    "promised_hours": promised, "processing_hours": processing,
    "transit_hours": transit, "delivery_hours": delivery_hours,
    "late": late, "transport_cost_inr": cost, "customer_rating": rating,
}).sort_values("order_time").reset_index(drop=True)

df.to_csv("data/shipments.csv", index=False)
print("Saved data/shipments.csv with", len(df), "rows")
print("Late rate:", round(df["late"].mean() * 100, 1), "percent")
print(df.groupby("hub")["late"].mean().round(3))
print(df.groupby("zone")["late"].mean().round(3))
print(df.groupby("shipping_mode")["late"].mean().round(3))
print(df["vehicle_type"].value_counts())
print(df[["delivery_hours", "transport_cost_inr", "weight_kg", "parcels"]].describe().round(1))
