# Experiments

Notes on the checks run while building the causal detector, in the order they were done. Each one is a question, a check, a result, and what changed because of it. All numbers come from the simulator with fixed seeds. Run the snippets from the repo root with the virtual environment active.

Setup shared by experiments 1 to 4. Run it first, since the later snippets use `data`, `scm` and `graph`:

```python
import pandas as pd

from causal_swarm.causal.graph import CausalGraph
from causal_swarm.causal.scm import SCM, build_fitting_data
from causal_swarm.simulation.swarm_env import Simulator, SimulatorConfig

config = SimulatorConfig(num_drone=4, num_timesteps=3000, dt=1.0, leader_id=0, seed=42)
data = build_fitting_data(Simulator(config).run(), config)
graph = CausalGraph()
scm = SCM.fit(graph, data)
```

## 1. Does the fitted model recover the simulator's true parameters?

**Question.** The constants in `swarm_env.py` generate the data, so a correct fit should get them back.

**Check.**

```python
for node in ["relative_velocity", "accel_magnitude", "signal_strength", "satellite_count",
             "battery_voltage", "gyro_z", "motor_rpm"]:
    print(node, scm.params[node])
```

**Result** (current code):

| Node | Fitted | True value in `swarm_env.py` |
|---|---|---|
| `relative_velocity` | beta 0.3000, sigma 0.0504 | gain 0.3, noise 0.05 |
| `accel_magnitude` | beta 0.1998, sigma 0.0499 | gain 0.2, noise 0.05 |
| `signal_strength` | alpha -29.96, beta -0.504, sigma 1.98 | -30, -0.5, noise 2.0 |
| `satellite_count` | alpha 9.99, sigma 1.00 | 10, noise 1.0 |
| `battery_voltage` | previous value 1.000, `current_draw` -0.0019, sigma 0.0099 | 1.0, -0.002, noise 0.01 |
| `gyro_z` | `heading_delta` 0.998, sigma 0.505 | 1.0, noise 0.5 |
| `motor_rpm` | `current_draw` 399.9, sigma 19.9 | 400, noise 20 |

**Not recovered.** In the first run, the two acceleration nodes gave small coefficients with inconsistent signs (experiments 2 and 4). `link_active` also did not recover: alpha 3.79 and beta -0.005 against the true 6.0 and 0.06. The cause is the data. In benign flight the model predicts a probability of 0.981 to 0.982 for every row, and the link is up in 98.1% of rows. The link almost never drops, so there are no observations near where the logistic curve is steep, and the slope cannot be estimated. This was left as it is. The score still works for a link drop, as experiment 5 shows.

## 2. Is the error size the same in all conditions? (`accel_x`)

**Question.** The anomaly score divides each error by one sigma per node. That is only fair if the typical error is the same size everywhere.

**Check.** Split the rows into 8 equal-count buckets of `formation_error` and compute the mean absolute error of the `accel_x` fit in each.

```python
def error_by_bucket(node, parents):
    p = SCM._fit_linear(data, node, parents)
    d = data[[node] + parents]
    error = (d[node] - (p["alpha"] + sum(b * d[k] for k, b in p["betas"].items()))).abs()
    return error.groupby(pd.qcut(data["formation_error"], 8), observed=True).mean()

before = error_by_bucket("accel_x", ["formation_error", "pos_x", "pos_y", "leader_pos_x", "leader_pos_y"])
print(before.iloc[0], before.iloc[-1], before.iloc[-1] / before.iloc[0])
```

**Result.** With the original parents (`formation_error`, `pos_x`, `pos_y`, `leader_pos_x`, `leader_pos_y`), the mean absolute error went from 0.207 in the lowest bucket to 0.380 in the highest, a ratio of 1.84. One sigma would be too small when the drone drifts far, so large but harmless drift would look anomalous.

**First diagnosis (revised in experiment 4).** `accel_x` is `accel_magnitude * cos(angle)`, where the angle depends on position. This looked nonlinear, so the conclusion was that a linear model could not capture it.

**Change.** Fit `accel_magnitude = hypot(accel_x, accel_y)` on `formation_error` instead, and keep `accel_x` and `accel_y` as raw telemetry columns.

**Re-check.**

```python
after = error_by_bucket("accel_magnitude", ["formation_error"])
print(after.iloc[0], after.iloc[-1], after.iloc[-1] / after.iloc[0])
print(scm.params["accel_magnitude"])
```

The buckets now run from 0.0398 to 0.0399 (ratio 1.00), with beta 0.1998 (true 0.2) and sigma 0.0499 (true 0.05).

## 3. How good is the fit for each node?

**Question.** Experiment 1 only looked at a handful of nodes. For every node, is the fitted sigma close to the true noise, and how much of the variation does the model explain?

**Check.** R squared, fitted sigma, and the mean residual per drone, before and after the changes below.

```python
def fit_report(node, parents):
    p = SCM._fit_linear(data, node, parents)
    d = data[[node] + parents].dropna()
    resid = d[node] - (p["alpha"] + sum(b * d[k] for k, b in p["betas"].items()))
    r2 = 1 - resid.var() / d[node].var()
    per_drone = resid.groupby(data.loc[resid.index, "drone_id"]).mean().round(2).to_dict()
    return round(p["sigma"], 2), round(r2, 3), per_drone

print(fit_report("pos_y", ["leader_pos_y"]))                # before
print(fit_report("pos_y", ["slot_y", "satellite_count"]))   # after
print(fit_report("gyro_z", ["mag_heading"]))                # before
print(fit_report("gyro_z", ["heading_delta"]))              # after
```

**Result.**

| Node | Before | After |
|---|---|---|
| `pos_x` sigma | 3.07 | 2.12 |
| `pos_y` sigma | 4.63 | 2.10 |
| `pos_y` mean residual per drone | -5.11, +5.01, +0.10 | -0.05, -0.05, +0.10 |
| `gyro_z` sigma | 5.04 (R squared 0.007) | 0.50 (R squared 0.990) |

The true tracking noise for the positions is about 2.1, and for `gyro_z` it is 0.5.

**Cause for the positions.** The three followers sit at different formation offsets (one 5 m left of the leader, one 5 m right), but the model had one shared intercept. Each drone's constant offset was counted as error. This inflated sigma and left every drone with a permanent bias. A 4 m GPS drift scored about 0.9 sigma before the fix and about 1.9 after.

**Cause for `gyro_z`.** It measures the turning rate, which is the heading change between timesteps. The model was using the heading value itself.

**Changes.**
- `build_fitting_data` adds `slot_x` and `slot_y`, which are the leader position plus the drone's assigned offset. They replace the leader position as the parent of `pos_x`, `pos_y` and `formation_error`.
- It also adds `heading_delta` (the shortest signed heading change per timestep) and `battery_voltage_lag1` (the previous `battery_voltage`). Both are computed per drone, so a shift never crosses a drone boundary. They are input nodes in the graph.

**Left as it is.** `formation_error` and `inter_drone_distance` have R squared of 0.013 and 0.027. Each is a distance, which is not a linear function of positions, so the model predicts roughly the average. Their scores mean "this distance is unusually large or small", not "this distance is consistent with the positions".

## 4. Revisiting experiment 2

**Question.** After the slot columns in experiment 3, does `accel_x` still have a problem?

**Check.**

```python
for label, parents in [("with leader_pos", ["formation_error", "pos_x", "pos_y", "leader_pos_x", "leader_pos_y"]),
                       ("with slot", ["formation_error", "pos_x", "pos_y", "slot_x", "slot_y"])]:
    p = SCM._fit_linear(data, "accel_x", parents)
    b = error_by_bucket("accel_x", parents)
    print(label, round(p["sigma"], 3), round(b.iloc[-1] / b.iloc[0], 2),
          {k: round(float(v), 3) for k, v in p["betas"].items()})
```

**Result.** With the slot columns, the ratio is 0.99, sigma is 0.036, and the coefficients are -0.2 on `pos_x`, +0.2 on `slot_x`, and 0 on everything else.

**Explanation.** The simulator computes `accel_x = -0.2 * (pos_x - slot_x) + noise * cos(angle)`. The `cos(angle)` only multiplies the noise, which is at most 0.05, so `accel_x` is linear in the deviation from the slot. The expected sigma is 0.05 / sqrt(2) = 0.035, which matches. The 1.84 in experiment 2 came from the missing per-drone offsets, the same cause as the position bias in experiment 3. The first diagnosis in experiment 2 was wrong.

`accel_magnitude` stays in the graph. It is one node with one parent and it recovers the gain directly. But fitting `accel_x` and `accel_y` would also have worked once the slot columns existed.

## 5. Do the anomaly scores catch attacks?

**Question.** If the model is fit on normal flight only, does the score rise during an attack, how soon, and at what cost in false alarms?

**Earlier check.** Adding 5 m to one drone's `pos_y` in the data raised the mean MCAI on those rows only from 1.94 to 2.37. The 99th percentile of benign MCAI was 3.33, so a single-row threshold would not catch it. That test changed one column only. A real spoof also moves `formation_error`, so the next check uses a real simulated attack.

**Setup.** Three runs with different seeds, so nothing is scored on data it was fit on:
- Train: seed 42, 3000 steps, no attacks (the `scm` above).
- Calibrate: seed 21, no attacks. The alarm threshold is the 99th percentile of MCAI on this run.
- Test: seed 11, with a GPS spoof on drone 3 (steps 1000 to 1300, 0.1 m per step) and a DoS flood on drone 2 (steps 1500 to 1600).

```python
from sklearn.metrics import roc_auc_score

from causal_swarm.causal.anomaly import centrality_weights, mcai, node_score
from causal_swarm.simulation.attacks import DoSFlood, GPSSpoof

weights = centrality_weights(graph)


def make(seed, attacks=(), steps=2000):
    cfg = SimulatorConfig(num_drone=4, num_timesteps=steps, dt=1.0, leader_id=0,
                          seed=seed, attacks=list(attacks))
    return build_fitting_data(Simulator(cfg).run(), cfg)


def scored(df):
    scores = node_score(scm, df)
    meta = df.loc[scores.index, ["drone_id", "timestep", "attack_type"]].copy()
    meta["mcai"] = mcai(scores, weights)
    meta["roll"] = meta.groupby("drone_id")["mcai"].transform(lambda x: x.rolling(20).mean())
    return scores, meta


_, calibration = scored(make(21))
threshold = calibration["mcai"].quantile(0.99)
threshold_roll = calibration["roll"].quantile(0.99)

attacks = [GPSSpoof(3, 1000, 1300, 0.1), DoSFlood(2, 1500, 1600)]
scores, test = scored(make(11, attacks))

none = test[test.attack_type == "none"]
print("false alarms:", (none["mcai"] > threshold).mean(), (none["roll"].dropna() > threshold_roll).mean())
for kind, start in [("gps_spoof", 1000), ("dos", 1500)]:
    rows = test[test.attack_type == kind]
    for column, limit in [("mcai", threshold), ("roll", threshold_roll)]:
        flagged = rows[rows[column] > limit]
        subset = test[test.attack_type.isin(["none", kind])].dropna(subset=["roll"])
        print(kind, column, "detected", round((rows[column] > limit).mean(), 3),
              "delay", flagged["timestep"].min() - start,
              "AUC", round(roc_auc_score(subset.attack_type == kind, subset[column]), 3))
    window = test.index[test.attack_type == kind]
    print(scores.loc[window].mean().sort_values(ascending=False).head(6).round(2).to_dict())
```

The scores are computed by `node_score`, weighted by `centrality_weights`, and summed per row by `mcai`. The rolling column is a 20-step mean per drone, a simple trend-style variant with its own threshold.

**Result.** The thresholds are 3.40 per row and 2.95 for the rolling mean.

| | Spoof (drone 3) | DoS (drone 2) |
|---|---|---|
| Rows flagged, per row | 78.3% | 100% |
| Rows flagged, rolling | 76.7% | 99.0% |
| First detection, per row | 64 steps after the start | 0 steps |
| First detection, rolling | 70 steps | 1 step |
| AUC, per row | 0.904 | 1.000 |
| AUC, rolling | 0.904 | 0.998 |

False alarms on unattacked rows are 0.7% per row and 1.6% for the rolling mean.

Mean node scores inside each attack window, with the top node scores listed:
- **Spoof:** `formation_error` 8.81, `pos_x` 7.21, `inter_drone_distance` 1.65. Every other node is 0.87 or lower. The spoof was along x, so `pos_y` stays normal.
- **DoS:** `latency` 99.3, `packet_rate` 17.8, `link_active` 7.0. Every other node is 1.05 or lower, including `signal_strength`.

**Reading the results.**
- The 22% of spoof rows that were not flagged are the early ones. A 64-step delay at 0.1 m per step is a 6.4 m drift, about 3 times the normal tracking noise (sigma 2.1). After that point nearly every row is flagged. The misses are all early rows, not random ones.
- The rolling mean did not help. It gave the same AUC and a slightly later first detection. What limits detection is how small a drift can be told apart from the drone's own slow wandering, not noise in the score.
- The DoS signature is `latency`, `packet_rate` and `link_active` abnormal while their parent `signal_strength` is normal. That points at the link, not at an upstream cause.
- The spoof signature is `pos_x` abnormal while its parents are normal. `formation_error` is also abnormal, but its parent `pos_x` is abnormal too, so its score is explained upstream. The children of `formation_error` (`relative_velocity`, `accel_magnitude`, `current_draw`) stay normal because they are consistent with it.
- `link_active` is a useful score even though its fitted slope is wrong. With a predicted probability of 0.98, an up link scores about 0.14 and a down link scores about 7.

**Limits.**
- One seed per run and one attack scenario. Detection delay depends on the drift rate: at 0.02 m per step, the same 6.4 m would take about 320 steps.
- The threshold comes from one calibration run.
- The DoS only changes the attacked drone's own link fields, not its neighbor's.
