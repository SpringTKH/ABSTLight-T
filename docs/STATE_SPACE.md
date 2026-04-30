# State Space Parameterization

## Single-Step Raw Observation Vector (20-Dimensional)

For each controlled intersection $i$, a single-timestep observation vector $\mathbf{o}_i^t \in \mathbb{R}^{20}$ is constructed in `get_flat_obs(tls_id)`. Its layout is as follows:

$$\mathbf{o}_i^t = \underbrace{[q_1, \ldots, q_K]}_{\text{queue features}} \;\|\; \underbrace{[w_1, \ldots, w_K]}_{\text{waiting features}} \;\|\; \underbrace{[\phi_0, \ldots, \phi_3]}_{\text{phase one-hot}}$$

Where $K = \lfloor(\text{obs\_dim} - 4) / 2\rfloor = 8$ lane slots, yielding $8 + 8 + 4 = 20$ dimensions.

**Normalisation.** Both queue and waiting features are clipped to $[0, 1]$ via conservative divisors to maintain numerical stability:

```python
# environment/traffic_env.py  ·  get_flat_obs()
obs[idx]              = np.clip(queue_length  / self.obs_queue_normalization,        0.0, 1.0)
obs[lane_slots + idx] = np.clip(waiting_time / self.obs_waiting_time_normalization, 0.0, 1.0)
```

| Feature            | TraCI Source               | Normalisation Divisor                    | Config Key                       |
| ------------------ | -------------------------- | ---------------------------------------- | -------------------------------- |
| Queue length $q_i$ | `getLastStepHaltingNumber` | `obs_queue_normalization = 50.0`         | `OBS_QUEUE_NORMALIZATION`        |
| Waiting time $w_i$ | `getWaitingTime`           | `obs_waiting_time_normalization = 300.0` | `OBS_WAITING_TIME_NORMALIZATION` |
| Phase one-hot      | `current_phase[tls_idx]`   | — (binary)                               | —                                |

**Phase encoding.** The active phase index $p \in \{0,1,2,3\}$ is encoded as a 4-dimensional one-hot vector appended at the tail of the observation:

```python
obs[self._single_obs_dim - phase_dim + phase_index] = 1.0
```

## Temporal Frame Stacking — T = 4 (80-Dimensional)

To capture temporal dynamics, the single-step observation is extended using a fixed-length **temporal frame stack** of depth $T = 4$. Per-intersection sliding windows are implemented via `collections.deque(maxlen=4)`, initialised in `get_flat_multi_obs()`:

```python
# environment/traffic_env.py  ·  get_flat_multi_obs()
self._obs_queues = [
    collections.deque(maxlen=self._obs_stack_size)   # maxlen = 4
    for _ in range(self.num_agents)
]
```

At each decision step, the most recent 20-D frame is appended to the deque (displacing the oldest frame), and the stacked observation is formed by concatenating all frames in **chronological order** (oldest → newest):

```python
self._obs_queues[idx].append(raw_obs.copy())

# Edge-case: pre-fill deque with the first frame if not yet full
while len(self._obs_queues[idx]) < self._obs_stack_size:
    self._obs_queues[idx].appendleft(raw_obs.copy())

# Concatenate oldest → newest to form the stacked observation
obs[idx] = np.concatenate(list(self._obs_queues[idx]), axis=0)
```

The resulting stacked observation tensor for all $N$ intersections has shape:

$$\mathbf{O}^t = \text{get\_flat\_multi\_obs()} \in \mathbb{R}^{N \times 80}$$

> **Implementation note.** Because the decision cadence is 10–13 simulation seconds (not 1 second), each of the $T = 4$ stacked frames represents a distinct agent decision epoch. The temporal window therefore spans approximately **40–52 seconds** of real traffic evolution.

## Context Flag Concatenation — 82-Dimensional (OOD Scenarios)

For out-of-distribution (OOD) robustness, the `MixedTrafficEnv` class (`environment/traffic_env_mixed.py`) extends the base observation with a **2-dimensional context tag** encoding the current environmental scenario. This is implemented via method override of `get_flat_multi_obs()`:

```python
# environment/traffic_env_mixed.py  ·  get_flat_multi_obs()
base_obs = super().get_flat_multi_obs()            # (N_agents, 80)
context  = self._get_context_tag()                 # (2,)
tiled    = np.tile(context, (self.num_agents, 1))  # (N_agents, 2)
return np.concatenate([base_obs, tiled], axis=1)   # (N_agents, 82)
```

**Context tag encoding.** The 2-D tag is a soft binary one-hot vector over three mutually exclusive scenario classes:

| Scenario        | `context_tag` | `rain_flag` | `accident_flag` |
| --------------- | ------------- | ----------- | --------------- |
| Sunny (nominal) | `[0.0, 0.0]`  | 0           | 0               |
| Rainy           | `[1.0, 0.0]`  | 1           | 0               |
| Accident        | `[0.0, 1.0]`  | 0           | 1               |

**Tensor shape trace — complete pipeline:**

```
traci.lane.*()                             →  scalar per lane
get_flat_obs(tls_id)                       →  (20,)           single-step raw obs
deque(maxlen=4)  ·  concatenate            →  (80,)           stacked temporal obs
get_flat_multi_obs() [base class]          →  (N_agents, 80)  all intersections
np.concatenate([base_obs, tiled], axis=1)  →  (N_agents, 82)  + context tag
```

The `total_obs_dim` property of `MixedTrafficEnv` exposes this as a constant:

```python
@property
def total_obs_dim(self) -> int:
    return self._STACK_OBS_DIM + self.CONTEXT_DIM   # 80 + 2 = 82
```
