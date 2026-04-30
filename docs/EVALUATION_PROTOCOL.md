# Metrics and Reproducibility Protocol

## Decision Interval and Safety Fallback Mechanism

SUMO advances in discrete steps of **1 second** (`traci.simulationStep()`). The agent does not act at every simulation step. A two-phase decision cadence is enforced at each agent decision point:

| Phase                 | Duration                    | Purpose                                               |
| --------------------- | --------------------------- | ----------------------------------------------------- |
| **Yellow transition** | `yellow_duration = 3 s`     | Mandatory clearance before any phase switch           |
| **Minimum green**     | `min_green_duration = 10 s` | Minimum interval before the next decision can be made |

Consequently, each agent decision step corresponds to a real-world interval of **10–13 simulation seconds**, depending on whether a phase change was requested.

```python
# Yellow transition applied only when at least one TLS changes phase
if changed_indices:
    for _ in range(self.yellow_duration):          # 3 SUMO ticks
        traci.simulationStep()
    traci.trafficlight.setPhase(tls_id, int(safe_actions[idx]))

# Minimum green hold — always executed
for _ in range(self.min_green_duration):           # 10 SUMO ticks
    traci.simulationStep()
```

A safety fallback mechanism (`_apply_safety_fallback`) prevents any single phase from being held for more than `SAFETY_FALLBACK_MAX_PHASE_HOLD = 60 s`, guarding against deadlock in edge cases.

## 3.8.1 Efficiency Metrics

### Average Waiting Time (`avg_waiting_time`)

**Definition.** The mean **accumulated waiting time** across all vehicles observed on controlled lanes during an episode, measured in seconds.

**TraCI API.**

```python
waiting_time_samples.append(
    float(traci.vehicle.getAccumulatedWaitingTime(veh_id))
)
```

`traci.vehicle.getAccumulatedWaitingTime(veh_id)` returns the total time (in seconds) that the vehicle has spent at a standstill (speed < 0.1 m/s) since entering the network.

```python
result["avg_waiting_time"] = mean(step_waiting_time_samples)
```

### Queue Length — Average, Maximum, and Variance

**Definition.** Queue length is the total number of halted vehicles (speed ≈ 0) across all lanes controlled by all intersections at a single simulation tick.

**TraCI API:**

```python
for lane_id in controlled_lanes:
    queue_length += float(traci.lane.getLastStepHaltingNumber(lane_id))
```

`traci.lane.getLastStepHaltingNumber(lane_id)` returns the count of vehicles on `lane_id` whose speed was below 0.1 m/s in the preceding simulation step.

| JSON Key                | Computation                  | Interpretation                                                    |
| ----------------------- | ---------------------------- | ----------------------------------------------------------------- |
| `avg_queue_length`      | `mean(step_queue_samples)`   | Mean total queue depth averaged over all decision steps           |
| `max_queue_length`      | `max(step_queue_samples)`    | Peak queue depth observed at any single tick during the episode   |
| `queue_length_variance` | `np.std(step_queue_samples)` | Standard deviation of queue depth (spread of congestion dynamics) |

```python
result["avg_queue_length"]      = mean(step_queue_samples)
result["max_queue_length"]      = float(max(step_queue_samples))
result["queue_length_variance"] = float(np.std(np.asarray(step_queue_samples, dtype=np.float32)))
```

### Time Loss (`time_loss`)

**Definition.** The mean **time loss** per vehicle per tick — the additional travel time incurred relative to the vehicle's free-flow travel time had it experienced no congestion or signal delay. Units: seconds.

**TraCI API:**

```python
time_loss_samples.append(float(traci.vehicle.getTimeLoss(veh_id)))
```

```python
result["time_loss"] = mean(step_time_loss_samples)
```

### Average Delay per Vehicle (`avg_delay_per_vehicle`)

**Definition.** A normalised speed-deficit measure, computed as the mean fractional speed loss relative to the lane's maximum allowed speed, averaged across all vehicles seen during the episode.

**Formula:**

$$\delta_i = \max\!\left(0,\; \frac{v_{\max,i} - v_i}{v_{\max,i}}\right)$$

where $v_i$ is `traci.vehicle.getSpeed(veh_id)` and $v_{\max,i}$ is `traci.vehicle.getAllowedSpeed(veh_id)`.

**TraCI APIs used:**

```python
speed         = float(traci.vehicle.getSpeed(veh_id))
allowed_speed = float(traci.vehicle.getAllowedSpeed(veh_id))
vehicle_delays[veh_id] = max(0.0, (allowed_speed - speed) / allowed_speed)
```

```python
per_vehicle_delay_means = [mean(samples) for samples in vehicle_delay_agg.values()]
result["avg_delay_per_vehicle"] = mean(per_vehicle_delay_means)
```

### Average Stop Time per Vehicle (`avg_stop_time_per_vehicle`)

**Definition.** The mean number of simulation seconds each unique vehicle spent fully stopped (speed ≤ 0.1 m/s) on a controlled lane during the episode.

```python
if speed <= 0.1:
    stopped_vehicle_ids.add(veh_id)

vehicle_stop_seconds[veh_id] = vehicle_stop_seconds.get(veh_id, 0) + 1
```

```python
stopped_seconds_total = sum(vehicle_stop_seconds.get(vid, 0) for vid in vehicle_seen_ids)
result["avg_stop_time_per_vehicle"] = stopped_seconds_total / len(vehicle_seen_ids)
```

### Average Incoming Road Speed (`avg_speed_incoming_road`)

**Definition.** The mean vehicle speed on all controlled lanes, averaged per tick and then across all ticks in the episode. Units: m/s.

**TraCI API:**

```python
lane_speeds.append(float(traci.lane.getLastStepMeanSpeed(lane_id)))
```

```python
result["avg_speed_incoming_road"] = mean(step_speed_samples)
```

## Throughput Metrics

### Arrived Vehicles (`arrived_vehicles`)

**Definition.** The total number of vehicles that successfully completed their routes and exited the network during the episode.

**TraCI API:**

```python
self.arrived_total += int(traci.simulation.getArrivedNumber())
```

`traci.simulation.getArrivedNumber()` returns the count of vehicles that arrived at their destination **in the current simulation step**.

```python
result["arrived_vehicles"] = int(arrived_total)
```

This metric directly reflects the network's vehicle throughput capacity.

### Intersection Throughput (`intersection_throughput_veh_per_step` and `intersection_throughput_veh_per_hour`)

**Definition.** The network-level vehicle throughput rate, derived from arrived vehicles and simulation duration.

**Formula:**

$$\text{throughput\_per\_step} = \frac{\text{arrived\_vehicles}}{\text{simulation\_seconds}}$$

$$\text{throughput\_veh\_per\_hour} = \text{throughput\_per\_step} \times 3600$$

```python
if simulation_seconds > 0:
    throughput_per_second = float(arrived_vehicles / simulation_seconds)
    result["intersection_throughput_veh_per_step"] = throughput_per_second
    result["intersection_throughput_veh_per_hour"] = float(throughput_per_second * 3600.0)
```

The `_veh_per_hour` variant is the primary reporting metric for cross-model comparison, as it normalises for episode duration.

### Green Wave Effectiveness (`green_wave_effectiveness`)

**Definition.** The fraction of vehicles currently on controlled lanes that are actively moving (speed > 0.1 m/s) at each simulation tick, averaged across all ticks in the episode.

**Formula per tick:**

$$\text{GWE}_t = \frac{\text{moving\_count}_t}{|\text{lane\_vehicle\_ids}_t|}$$

```python
moving_count = sum(1 for veh_id in lane_vehicle_ids if traci.vehicle.getSpeed(veh_id) > 0.1)
green_wave_effectiveness = float(moving_count / len(lane_vehicle_ids))
```

```python
result["green_wave_effectiveness"] = mean(step_green_samples)
```

### Phase Switch Count (`phase_switch_count`)

**Definition.** The total number of traffic signal phase changes across all controlled intersections during the episode, normalised by the number of agents.

```python
if current_phase != previous_phase_states[tls_id]:
    tracker.register_phase_switches(1)
```

```python
result["phase_switch_count"] = float(self.phase_switch_count / self.num_agents)
```

## Safety and OOD Stability Metrics

### Teleportation by Collisions (`teleportation_by_collisions`)

**Definition.** The number of simulation steps in which a **collision-induced teleportation** event occurred:

$$\text{teleportation\_by\_collisions} = \min(\text{getEndingTeleportNumber},\; \text{getCollidingVehiclesNumber})$$

**TraCI APIs:**

```python
teleport_count   = int(traci.simulation.getEndingTeleportNumber())
collision_count  = int(traci.simulation.getCollidingVehiclesNumber())
collision_teleport_count = int(min(teleport_count, collision_count))
```

```python
self.teleportation_by_collisions += int(metrics["collision_teleport_count"])
result["teleportation_by_collisions"] = int(self.teleportation_by_collisions)
```

### Safety Fallback Count (`safety_fallback_count`)

**Definition.** The number of decision steps during the episode at which the `_apply_safety_fallback()` mechanism overrode the agent's chosen action to prevent a phase being held beyond `SAFETY_FALLBACK_MAX_PHASE_HOLD = 60 s`.

```python
result["safety_fallback_count"] = int(safety_fallback_count)
# Where:
safety_fallback_count = int(latest_info.get("safety_fallback_count", 0))
```

A non-zero value indicates episodes in which the safety mechanism intervened.

## Environmental Impact Metrics

These two metrics are recorded during evaluation but are **not included in the `TARGET_METRICS` list used by `run_seed_suite.py`** for cross-model statistical comparison.

### CO₂ Emissions (`co2_emissions_total_mg`)

**Definition.** The total CO₂ emitted by all vehicles in the network across all simulation ticks of the episode, in milligrams.

```python
for veh_id in traci.vehicle.getIDList():
    co2_mg += float(traci.vehicle.getCO2Emission(veh_id))
```

```python
self.co2_total_mg += float(metrics["co2_mg"])
result["co2_emissions_total_mg"] = float(self.co2_total_mg)
```

### Fuel Consumption (`fuel_consumption_total_litre`)

**Definition.** The total fuel consumed by all vehicles in the network across the episode, converted from SUMO's native mg/s units to litres using a petrol density of 0.745 kg/L.

```python
for veh_id in traci.vehicle.getIDList():
    fuel_mg += float(traci.vehicle.getFuelConsumption(veh_id))

# SUMO fuel is reported in mg/s; convert to litres with gasoline density.
# litres = mg / (1e6 mg/kg × 0.745 kg/L)
fuel_litre = float(fuel_mg / (1_000_000.0 * 0.745))
```

```python
self.fuel_total_litre += float(metrics["fuel_litre"])
result["fuel_consumption_total_litre"] = float(self.fuel_total_litre)
```

## Cross-Seed Statistical Reporting

The primary evaluation protocol runs each model–scenario combination across **5 fixed random seeds** (`[42, 100, 1234, 2026, 8888]`) using `run_seed_suite.py`. For the **7 headline metrics** defined in `TARGET_METRICS`, the script computes the **mean** and **population standard deviation** across seeds:

```python
TARGET_METRICS = [
    "arrived_vehicles",
    "avg_delay_per_vehicle",
    "max_queue_length",
    "queue_length_variance",
    "time_loss",
    "phase_switch_count",
    "teleportation_by_collisions",
]

statistical_summary[metric] = {
    "mean":    float(statistics.fmean(values)),
    "std_dev": float(statistics.pstdev(values)),
}
```

The consolidated output is written to `results/<scenario>/final_summary_<route>.json`, with the following top-level structure:

```json
{
  "route_file": "sumo_files/osm.peak.rou.xml",
  "seeds_expected": [42, 100, 1234, 2026, 8888],
  "models": {
    "ABSTLight_Base": {
      "runs_by_seed": [ ... ],
      "statistical_summary": {
        "arrived_vehicles":        { "mean": ..., "std_dev": ... },
        "avg_delay_per_vehicle":   { "mean": ..., "std_dev": ... },
        "max_queue_length":        { "mean": ..., "std_dev": ... },
        "queue_length_variance":   { "mean": ..., "std_dev": ... },
        "time_loss":               { "mean": ..., "std_dev": ... },
        "phase_switch_count":      { "mean": ..., "std_dev": ... },
        "teleportation_by_collisions": { "mean": ..., "std_dev": ... }
      }
    },
    "ABSTLight_FT": { ... },
    "Actuated":     { ... },
    "Fixed-Time":   { ... }
  }
}
```

**Model label mapping** (from `run_seed_suite.py`):

| Internal `model_type` key | Summary label    | Test mode   |
| ------------------------- | ---------------- | ----------- |
| `ABSTLIGHT`               | `ABSTLight_Base` | `--model 2` |
| `ABSTLIGHT_FINETUNED`     | `ABSTLight_FT`   | `--model 4` |
| `SUMO_DEFAULT`            | `Actuated`       | `--model 1` |
| `SUMO_FIXED_TIME`         | `Fixed-Time`     | `--model 3` |

## 3.8.6 Complete Metric Reference Table

| JSON Key                               | Unit           | Data Type | In `TARGET_METRICS`? | Description                                                                         |
| -------------------------------------- | -------------- | --------- | -------------------- | ----------------------------------------------------------------------------------- |
| `simulation_seconds`                   | seconds        | int       | No                   | Total SUMO simulation ticks elapsed in the episode                                  |
| `decision_steps`                       | steps          | int       | No                   | Number of agent decision cycles (≈ simulation_seconds / 10–13)                      |
| `arrived_vehicles`                     | vehicles       | int       | **Yes**              | Vehicles that completed their routes and exited the network                         |
| `safety_fallback_count`                | triggers       | int       | No                   | Safety fallback overrides (phase held > 60 s)                                       |
| `avg_delay_per_vehicle`                | fraction [0,1] | float     | **Yes**              | Mean normalised speed deficit: $(v_{\max} - v) / v_{\max}$                          |
| `avg_speed_incoming_road`              | m/s            | float     | No                   | Mean lane speed across controlled lanes and all ticks                               |
| `avg_queue_length`                     | vehicles       | float     | No                   | Mean total halting vehicles across all ticks                                        |
| `max_queue_length`                     | vehicles       | float     | **Yes**              | Maximum total halting vehicles at any single tick                                   |
| `queue_length_variance`                | vehicles (σ)   | float     | **Yes**              | Std dev of per-tick queue depth                                                     |
| `avg_waiting_time`                     | seconds        | float     | No                   | Mean accumulated waiting time per vehicle per tick                                  |
| `time_loss`                            | seconds        | float     | **Yes**              | Mean cumulative time loss per vehicle per tick                                      |
| `phase_switch_count`                   | switches/agent | float     | **Yes**              | Phase changes per intersection, normalised by number of agents                      |
| `avg_stop_time_per_vehicle`            | seconds        | float     | No                   | Mean simulation seconds each unique vehicle spent fully stopped                     |
| `teleportation_by_collisions`          | events         | int       | **Yes**              | Collision-induced teleportation events: $\min(\text{teleports}, \text{collisions})$ |
| `fuel_consumption_total_litre`         | litres         | float     | No                   | Total fuel consumed by all network vehicles across the episode                      |
| `intersection_throughput_veh_per_step` | veh/s          | float     | No                   | Arrived vehicles per simulation second                                              |
| `intersection_throughput_veh_per_hour` | veh/h          | float     | No                   | Arrived vehicles normalised to a per-hour rate                                      |
| `green_wave_effectiveness`             | fraction [0,1] | float     | No                   | Fraction of vehicles on controlled lanes that are moving                            |
| `co2_emissions_total_mg`               | mg             | float     | No                   | Network-wide CO₂ emissions (HBEFA model, all vehicles)                              |
