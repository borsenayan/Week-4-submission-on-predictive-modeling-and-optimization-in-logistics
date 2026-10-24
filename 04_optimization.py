"""Step 4: use the forecasts to optimise staffing, expediting and routes."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.spatial.distance import cdist

os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 20)
results = {}

# Step O1: Staffing plan for the festive peak with integer programming
print("### O1")
vol = pd.read_csv("data/volume_forecast.csv", parse_dates=["date"])
week = vol[(vol["date"] >= "2025-11-03") & (vol["date"] <= "2025-11-09")]
hubs = ["Pune", "Nagpur", "Sambhajinagar", "Nashik"]
raw = week.groupby("hub")["forecast"].mean().reindex(hubs).values
SAFETY = 0.10       # the forecast runs about 8 percent low so add a margin
forecast = raw * (1 + SAFETY)
base_hours = np.array([8, 9, 10, 17])             # hub processing speed from the EDA
capacity = 32 / base_hours                          # shipments one picker handles a day
need = forecast / capacity
pool = int(np.ceil(need.sum() * 0.9))               # 10 percent fewer pickers than needed
print("Forecast shipments per day:",
      dict(zip(hubs, forecast.round(1).tolist())))
print("Pickers needed for full coverage:", need.round(1).tolist(),
      "Pool available:", pool)

# Current practice: share the pickers in proportion to volume
share = forecast / forecast.sum() * pool
naive = np.floor(share).astype(int)
for i in np.argsort(share - naive)[::-1][: pool - naive.sum()]:
    naive[i] += 1

# Optimised plan: maximise the lowest coverage ratio across hubs
# variables: staff at each hub (integer) and z (the coverage ratio to maximise)
c = np.r_[np.zeros(4), -1.0]
rows = np.c_[-np.diag(capacity), forecast]          # z * volume <= capacity * staff
cons = [LinearConstraint(rows, -np.inf, 0),
        LinearConstraint(np.r_[np.ones(4), 0].reshape(1, -1), 0, pool)]
res = milp(c, constraints=cons, integrality=np.r_[np.ones(4), 0],
           bounds=Bounds(np.r_[np.full(4, 2.0), 0], np.r_[np.full(4, pool), 5]))
best = np.round(res.x[:4]).astype(int)

def coverage(staff):
    return capacity * staff / forecast

plan = pd.DataFrame({"hub": hubs, "volume": forecast.round(1),
                     "naive staff": naive, "naive coverage": coverage(naive).round(2),
                     "optimal staff": best,
                     "optimal coverage": coverage(best).round(2)}).set_index("hub")
print(plan)
short_naive = np.clip(forecast - capacity * naive, 0, None).sum()
short_best = np.clip(forecast - capacity * best, 0, None).sum()
print("Shipments left over per day, naive:", round(short_naive, 1),
      "optimised:", round(short_best, 1))
results["staff_plan"] = plan.reset_index().to_dict("records")
results["pool"] = pool
results["short_naive"] = round(float(short_naive), 1)
results["short_best"] = round(float(short_best), 1)
results["min_cov_naive"] = round(float(coverage(naive).min()), 2)
results["min_cov_best"] = round(float(coverage(best).min()), 2)

# Step O2: Predict late orders and expedite the risky ones
print("### O2")
mt = pd.read_csv("data/model_table.csv")
pred = pd.read_csv("data/test_predictions.csv")
t = mt.merge(pred, on="shipment_id")
VALUE_PER_LATE = 150        # assumed rupee cost of one late delivery
EXPEDITE_COST = 25          # assumed rupee cost of expediting one shipment
SPEEDUP = 0.35              # assumed cut in processing time when expedited

base_late = int(t["late"].sum())
rows = []
for buffer in range(-8, 9, 2):
    flagged = t["pred"] > t["promised_hours"] + buffer
    processing = np.where(flagged, t["processing_hours"] * (1 - SPEEDUP),
                          t["processing_hours"])
    new_hours = processing + t["transit_hours"]
    new_late = int((new_hours > t["promised_hours"]).sum())
    avoided = base_late - new_late
    net = avoided * VALUE_PER_LATE - int(flagged.sum()) * EXPEDITE_COST
    rows.append({"buffer_h": buffer, "flagged": int(flagged.sum()),
                 "late_avoided": avoided, "net_benefit": net})
sweep = pd.DataFrame(rows).set_index("buffer_h")
print("Late shipments without action:", base_late, "of", len(t))
print(sweep)
best_buffer = int(sweep["net_benefit"].idxmax())
print("Best buffer:", best_buffer, "hours with net benefit of Rs",
      int(sweep["net_benefit"].max()))
results["sweep"] = sweep.reset_index().to_dict("records")
results["best_buffer"] = best_buffer
results["base_late"] = base_late
results["n_test"] = len(t)

# Step O3: Route planning with the savings algorithm
print("### O3")
def savings_routes(dist, demand, cap):
    """Clarke and Wright savings algorithm with a vehicle capacity."""
    n = len(demand)
    routes = {i: [i] for i in range(1, n)}
    load = {i: demand[i] for i in routes}
    where = {i: i for i in routes}
    pairs = sorted(((dist[0][i] + dist[0][j] - dist[i][j], i, j)
                    for i in range(1, n) for j in range(i + 1, n)), reverse=True)
    for _, i, j in pairs:
        ri, rj = where[i], where[j]
        if ri == rj or load[ri] + load[rj] > cap:
            continue
        a, b = routes[ri], routes[rj]
        if a[-1] == i and b[0] == j:
            new = a + b
        elif a[0] == i and b[-1] == j:
            new = b + a
        elif a[-1] == i and b[-1] == j:
            new = a + b[::-1]
        elif a[0] == i and b[0] == j:
            new = a[::-1] + b
        else:
            continue
        routes[ri] = new
        load[ri] += load[rj]
        for k in b:
            where[k] = ri
        del routes[rj], load[rj]
    return list(routes.values())

def route_length(route, dist):
    path = [0] + route + [0]
    return sum(dist[a][b] for a, b in zip(path[:-1], path[1:]))

def two_opt(route, dist):
    """Reverse parts of a route while that makes it shorter."""
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                new = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                if route_length(new, dist) < route_length(route, dist) - 1e-9:
                    route, improved = new, True
    return route

rng = np.random.default_rng(11)
n_stops, capacity_kg = 30, 400
angle = rng.uniform(0, 2 * np.pi, n_stops)
radius = rng.uniform(3, 40, n_stops)
xy = np.vstack([[0, 0], np.c_[radius * np.cos(angle), radius * np.sin(angle)]])
demand = np.r_[0, rng.integers(20, 120, n_stops)]
dist = cdist(xy, xy)

# Current practice: fill vans in the order of the angle around the hub
order = list(np.argsort(angle) + 1)
sweep_routes, current, load = [], [], 0
for stop in order:
    if load + demand[stop] > capacity_kg:
        sweep_routes.append(current)
        current, load = [], 0
    current.append(stop)
    load += demand[stop]
sweep_routes.append(current)

optimised = [two_opt(r, dist) for r in savings_routes(dist, demand, capacity_kg)]
km_sweep = sum(route_length(r, dist) for r in sweep_routes)
km_opt = sum(route_length(r, dist) for r in optimised)
COST_PER_KM = 9
print("Stops:", n_stops, "Total load:", int(demand.sum()), "kg", "Van capacity:",
      capacity_kg, "kg")
print("Sweep plan:", len(sweep_routes), "vans,", round(km_sweep, 1), "km")
print("Optimised plan:", len(optimised), "vans,", round(km_opt, 1), "km")
print("Distance saved:", f"{1 - km_opt / km_sweep:.1%}",
      "Cost saved per day: Rs", round((km_sweep - km_opt) * COST_PER_KM))
results.update({"stops": n_stops, "km_sweep": round(km_sweep, 1),
                "km_opt": round(km_opt, 1), "vans_sweep": len(sweep_routes),
                "vans_opt": len(optimised),
                "km_saved_pct": round((1 - km_opt / km_sweep) * 100, 1),
                "route_saving_rs": round((km_sweep - km_opt) * COST_PER_KM)})

# Step Z: Charts and saved results
fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(4)
ax.bar(x - 0.2, coverage(naive), 0.4, label="Share by volume", color="#BFD3EE")
ax.bar(x + 0.2, coverage(best), 0.4, label="Optimised", color="#1F3864")
ax.axhline(1, color="#C0392B", ls="--", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(hubs)
ax.set_ylabel("Capacity divided by forecast volume")
ax.set_title("Picker coverage by hub in the festive peak week")
ax.legend()
plt.tight_layout(); plt.savefig("figures/o01_staffing.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(sweep.index, sweep["net_benefit"] / 1000, marker="o", color="#2E5597")
ax.axvline(best_buffer, color="#C0392B", ls="--", lw=1)
ax.set_xlabel("Buffer in hours (flag when predicted time exceeds promise plus buffer)")
ax.set_ylabel("Net benefit (Rs thousand)")
ax.set_title("Net benefit of expediting by flagging threshold")
plt.tight_layout(); plt.savefig("figures/o02_expedite.png", dpi=170); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
for ax, routes, title in [(axes[0], sweep_routes, f"Sweep plan: {km_sweep:.0f} km"),
                          (axes[1], optimised, f"Optimised plan: {km_opt:.0f} km")]:
    for k, r in enumerate(routes):
        pts = xy[[0] + r + [0]]
        ax.plot(pts[:, 0], pts[:, 1], marker="o", ms=3, lw=1.3, color=f"C{k % 10}")
    ax.scatter(*xy[0], marker="s", s=70, color="black", zorder=5)
    ax.set_title(title)
    ax.set_aspect("equal")
fig.suptitle("Van routes from the hub (square) to 30 stops")
plt.tight_layout(); plt.savefig("figures/o03_routes.png", dpi=170); plt.close()
with open("outputs/opt_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
