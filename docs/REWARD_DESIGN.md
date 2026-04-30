# Reward Mechanism and Safety

## Base Reward Function (Standard Scenarios)

The per-intersection reward $r_i^t$ is a four-component linear combination computed in `compute_reward()` (`environment/traffic_env.py`). Each component is designed to incentivise a specific traffic control objective:

$$r_i^t = w_T \cdot T_i + w_P \cdot \frac{P_i}{C} + \text{clip}(w_I \cdot \delta_\text{ins},\; \rho_I) + w_\phi \cdot \mathbb{1}[\phi_i\text{ changed}]$$

```python
rewards[idx] = (
    self.reward_weights['throughput']        * throughput            # w_T · T_i
    + self.reward_weights['pressure']        * normalized_pressure   # w_P · P_i/C
    + insertion_component                                            # clipped insertion
    + phase_change_component                                         # phase penalty
)
```

**Component definitions and default weights:**

| Component            | Symbol                                               | Formula                                                                      | Default Weight | Config Key                        |
| -------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------- | -------------- | --------------------------------- |
| Throughput reward    | $w_T \cdot T_i$                                      | Destination arrival rate based on induction-loop vehicle count (speed-gated) | **+1.0**       | `REWARD_THROUGHPUT_WEIGHT`        |
| Pressure penalty     | $w_P \cdot P_i/C$                                    | $(V_\text{in} - V_\text{out})^+ / C$, $C=50$                                 | **−0.5**       | `REWARD_PRESSURE_WEIGHT`          |
| Insertion penalty    | $\text{clip}(w_I \cdot \delta_\text{ins},\, \rho_I)$ | Per-agent pending vehicles / $C$                                             | **−1.0**       | `REWARD_INSERTION_PENALTY_WEIGHT` |
| Phase-change penalty | $w_\phi$                                             | Applied once per decision step if phase changed                              | **−1.0**       | `phase_change_penalty`            |

**Pressure normalisation** divides by capacity constant $C = 50$ (`PRESSURE_NORMALIZATION_CAPACITY`) to keep the scale comparable to the throughput signal.

**Throughput gating** prevents rewarding stationary vehicles sitting atop induction loops; a vehicle is only counted if `mean_speed > speed_threshold` (default `2.0 m/s` for sunny scenarios):

```python
mean_speed = float(traci.inductionloop.getLastStepMeanSpeed(loop_id))
if mean_speed <= self.speed_threshold:
    continue
passed += float(traci.inductionloop.getLastStepVehicleNumber(loop_id))
```

**Insertion penalty clipping.** The insertion penalty is hard-clipped to prevent it from dominating the reward signal during early training when many vehicles are queued outside the network:

```python
insertion_component = self.reward_weights['insertion_penalty'] * insertion_signal
insertion_component = max(insertion_component, self.insertion_penalty_clip)
```

The default clip floor is `INSERTION_PENALTY_CLIP = -30.0` (base environment).

## Scenario-Aware Reward Shaping (OOD Accident Scenarios)

For accident episodes, `MixedTrafficEnv.compute_reward()` applies **spatially localised** reward weights that differentiate between the affected TLS (directly serving the blocked edge) and unaffected junctions. This prevents the parked-vehicle congestion signal from being mistakenly penalised.

**Affected TLS (accident junction) — relaxed weights:**

| Component            | Accident Weight | Standard Weight | Rationale                                              |
| -------------------- | --------------- | --------------- | ------------------------------------------------------ |
| Throughput           | **+1.5**        | +1.0            | Amplify any vehicle movement past the obstruction      |
| Pressure             | **−0.1**        | −0.5            | Tolerate unavoidable queue on blocked lane             |
| Insertion            | **−0.3**        | −1.0            | Softer penalty; network-wide congestion is expected    |
| Insertion clip floor | **−5.0**        | −30.0           | Tighter cap to bound signal magnitude                  |
| Phase-change         | **−0.2**        | −1.0            | Encourage aggressive switching to clear adjacent lanes |

**Unaffected junctions — standard weights with hard phase penalty:**

Junctions not serving the blocked edge retain the standard weights (`throughput=1.0`, `pressure=-0.5`, `insertion=-1.0`) but are assigned a fixed `phase_w = -1.0` regardless of the scenario's `phase_change_penalty` parameter, ensuring the rest of the network does not idle during an incident.

```python
# environment/traffic_env_mixed.py  ·  compute_reward()
if is_accident_tls:
    throughput_w, pressure_w, insertion_w, ins_clip, phase_w = 1.5, -0.1, -0.3, -5.0, self.phase_change_penalty  # -0.2
else:
    throughput_w, pressure_w, insertion_w, ins_clip, phase_w = (
        self.reward_weights['throughput'], self.reward_weights['pressure'],
        self.reward_weights['insertion_penalty'], self.insertion_penalty_clip,
        -1.0   # standard tight phase-change penalty
    )
```

**Scenario-dependent phase-change penalty.** The phase-change penalty $w_\phi$ is also differentiated by scenario type via `_SCENARIO_PARAMS`:

| Scenario                   | `phase_change_penalty` | Speed Threshold |
| -------------------------- | ---------------------- | --------------- |
| Sunny (nominal)            | −1.0                   | 2.0 m/s         |
| Rainy                      | −0.3                   | 1.4 m/s         |
| Accident (at affected TLS) | −0.2                   | 0.5 m/s         |

## Deadlock-Prevention Penalty

A supplementary penalty of $-2.0$ is applied to the accident TLS when spill-over congestion is detected on the **open** lanes of the blocked edge — excluding the blocked lane itself, which is permanently congested by the parked vehicle and therefore outside agent control:

```python
# Condition: > 80% of vehicles on OPEN lanes of the blocked edge are halted
if open_total > 0 and (open_halting / open_total) > 0.8:
    rewards[idx] -= 2.0
```

This signal specifically penalises controllable congestion propagation, providing a learnable gradient without injecting noise from the uncontrollable blocked lane.
