"""
traffic_env_mixed.py  —  Mixed/Universal Traffic Environment
=============================================================

Extends TrafficEnvironment via strict OOP inheritance to add:

  1. **Domain Randomization** — each episode independently samples one of
     three scenarios (Sunny, Rainy, Accident) from a configurable probability
     distribution.

  2. **Dynamic Reward Adaptation** — speed_threshold and phase_change_penalty
     are updated at the start of every episode to maintain MDP consistency
     under each condition.

  3. **82-Dimensional Observations** — appends a static 2-D context tag to
     the 80-D stacked observation produced by the base class, without
     breaking the ``obs_dim % obs_stack_size == 0`` constraint.

State vector layout  (per intersection, per step)
--------------------------------------------------
  indices  0 – 79   80-D stacked temporal observation
                    (produced by super().get_flat_multi_obs())
  index   80        rain_flag      {0.0, 1.0}
  index   81        accident_flag  {0.0, 1.0}

Context encoding
----------------
  Sunny / Normal  →  [0, 0]
  Rainy           →  [1, 0]
  Accident        →  [0, 1]

Key design decisions
--------------------
* ``super().__init__()`` always receives ``obs_dim=80`` so the base-class
  stacking check (80 % 4 == 0) passes.  The 2-D tag is appended *after*
  stacking, not inside the stacking window.

* ``total_obs_dim`` (= 82) is the value that must be passed to
  ``ABSTLightAgent(obs_dim=...)`` and ``ReplayBuffer(state_shape=...)``.

* The original ``environment/traffic_env.py`` is **not modified**.

Route files used per scenario
------------------------------
  Sunny    : osm.moderate.rou.xml  /  osm.peak.rou.xml  (50:50 random)
  Rainy    : osm.rain_m.rou.xml    /  osm.rain_h.rou.xml  (50:50 random)
  Accident : osm.acc1.rou.xml  /  osm.acc2.rou.xml  /
             osm.acc3.rou.xml  /  osm.acc4.rou.xml  (random selection;
             each file is osm.moderate.rou.xml + one statically parked
             truck that blocks a specific lane; enable_accident not used)
             since the accident is pre-baked into the route file)

Reward parameters per scenario
--------------------------------
  Sunny    : speed_threshold=2.0,  phase_change_penalty=-1.0
  Rainy    : speed_threshold=1.4,  phase_change_penalty=-0.3
  Accident : speed_threshold=0.5,  phase_change_penalty=-0.2
"""

import traci
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from environment.traffic_env import TrafficEnvironment


# ============================================================================
# Module-level constants
# ============================================================================

# Route files (relative to project root)
_ROUTES_SUNNY: List[str] = [
    "sumo_files/osm.moderate.rou.xml",
    "sumo_files/osm.peak.rou.xml",
]
_ROUTES_RAINY: List[str] = [
    "sumo_files/osm.rain_m.rou.xml",
    "sumo_files/osm.rain_h.rou.xml",
]
_ROUTES_ACCIDENT: List[str] = [
    "sumo_files/osm.acc1.rou.xml",  # Jalan Tun Tan Cheng Lock (3-lane, middle blocked)
    "sumo_files/osm.acc2.rou.xml",  # Jalan Yap Ah Loy        (3-lane, middle blocked)
    "sumo_files/osm.acc3.rou.xml",  # Jalan Petaling           (2-lane, right  blocked)
    "sumo_files/osm.acc4.rou.xml",  # Jalan Tun Tan Cheng Lock seg-2 (2-lane, right blocked)
]

# Reward parameters per scenario
_SCENARIO_PARAMS: Dict[str, Dict[str, float]] = {
    "sunny":    {"speed_threshold": 2.0, "phase_change_penalty": -1.0},
    "rainy":    {"speed_threshold": 1.4, "phase_change_penalty": -0.3},
    "accident": {"speed_threshold": 0.5, "phase_change_penalty": -0.2},
}

# Default scenario probability distribution
_DEFAULT_PROBS: Dict[str, float] = {
    "sunny":    0.50,
    "rainy":    0.30,
    "accident": 0.20,
}

_VALID_SCENARIOS = frozenset(_DEFAULT_PROBS.keys())

# Maps accident route-file stem → (blocked_edge_id, blocked_lane_id).
# blocked_lane_id is the lane where the parked accident_veh stops.
# Values are derived directly from gen_accidents_routes.py stop_lane config.
_ACCIDENT_EDGE_MAP: Dict[str, Tuple[str, str]] = {
    "osm.acc1": ("135488882#0", "135488882#0_1"),   # Jalan Tun Tan Cheng Lock (3-lane)
    "osm.acc2": ("771688494#0", "771688494#0_1"),   # Jalan Yap Ah Loy         (3-lane)
    "osm.acc3": ("135623338#0", "135623338#0_1"),   # Jalan Petaling           (2-lane)
    "osm.acc4": ("28215290#0",  "28215290#0_1"),    # Jalan Tun Tan Cheng Lock seg-2 (2-lane)
}


# ============================================================================
# MixedTrafficEnv
# ============================================================================

class MixedTrafficEnv(TrafficEnvironment):
    """
    Mixed environment that randomly samples a traffic scenario each episode.

    Args
    ----
    sumo_config : str
        Path to the ``.sumocfg`` file (forwarded to ``TrafficEnvironment``).
    use_gui : bool
        Whether to launch SUMO with a GUI.
    num_actions : int
        Number of discrete traffic-signal phases.
    obs_stack_size : int
        Temporal frame-stacking window (default 4).  Must divide 80 evenly.
    yellow_duration : int
        Yellow-phase duration in seconds.
    min_green_duration : int
        Minimum green-phase duration in seconds.
    reward_weights : dict | None
        Overrides for the four reward-component weights.
    max_steps : int
        Maximum simulation steps per episode.
    pressure_normalization_capacity : float
        Capacity divisor for pressure normalisation.
    insertion_penalty_clip : float
        Minimum (most negative) value for the insertion-delay penalty.
    obs_queue_normalization : float
        Divisor applied to raw queue counts in ``get_flat_obs()``.
    obs_waiting_time_normalization : float
        Divisor applied to raw waiting-time values in ``get_flat_obs()``.
    scenario_probs : dict | None
        Probability mass for each scenario key (``'sunny'``, ``'rainy'``,
        ``'accident'``).  Values must sum to 1.0.
        Defaults to ``{'sunny': 0.50, 'rainy': 0.30, 'accident': 0.20}``.

    Properties
    ----------
    total_obs_dim : int
        Total per-intersection observation dimension seen by the agent: **82**.
        Use this value for ``ABSTLightAgent(obs_dim=...)`` and
        ``ReplayBuffer(state_shape=(num_agents, total_obs_dim))``.

    current_scenario : str
        The scenario sampled at the most recent ``reset()`` call.
        One of ``'sunny'``, ``'rainy'``, ``'accident'``.
    """

    CONTEXT_DIM: int = 2   # [rain_flag, accident_flag]
    _STACK_OBS_DIM: int = 80

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        sumo_config: str,
        use_gui: bool = False,
        num_actions: int = 4,
        obs_stack_size: int = 4,
        yellow_duration: int = 3,
        min_green_duration: int = 10,
        reward_weights: Optional[Dict[str, float]] = None,
        max_steps: int = 3600,
        pressure_normalization_capacity: float = 50.0,
        insertion_penalty_clip: float = -30.0,
        obs_queue_normalization: float = 50.0,
        obs_waiting_time_normalization: float = 300.0,
        scenario_probs: Optional[Dict[str, float]] = None,
        dynamic_accident_activation: bool = False,
        accident_vehicle_id: str = "accident_veh",
        sumo_seed: Optional[int] = None,
    ):
        # ---- Validate and store scenario probability distribution FIRST ----
        self._scenario_probs: Dict[str, float] = self._validate_probs(scenario_probs)

        # ---- Initialise current scenario placeholder ----
        # (will be set properly on each reset())
        self.current_scenario: str = "sunny"

        # ---- Accident spatial-shaping state (populated per episode) ----
        self.current_blocked_edge:  str       = ""   # edge ID, e.g. "135488882#0"
        self.current_blocked_lane:  str       = ""   # lane ID, e.g. "135488882#0_1"
        self._accident_tls_indices: List[int] = []   # TLS indices controlling the blocked edge

        # ---- Optional runtime accident activation state ----
        self.dynamic_accident_activation = bool(dynamic_accident_activation)
        self.accident_vehicle_id = str(accident_vehicle_id)
        self._accident_triggered_runtime = False
        self._accident_route_has_metadata = False
        self._accident_activation_step: Optional[int] = None
        self._accident_stop_lane: str = ""
        self._accident_stop_start_pos: float = 0.0
        self._accident_stop_end_pos: float = 0.0

        # ---- Delegate to parent with obs_dim=80  ────────────────────────
        #
        # CRITICAL: we always pass obs_dim=80 here regardless of the agent's
        # total_obs_dim (82).  This ensures the stacking divisibility check
        # inside TrafficEnvironment.__init__() (obs_dim % obs_stack_size == 0)
        # evaluates as  80 % 4 == 0  and passes cleanly.
        #
        # The 2-D context tag is appended in get_flat_multi_obs() AFTER the
        # base-class stacking logic, so it is never counted against the stack.
        # ─────────────────────────────────────────────────────────────────
        super().__init__(
            sumo_config=sumo_config,
            use_gui=use_gui,
            route_file=None,              # managed per-episode in _sample_and_apply_scenario()
            num_actions=num_actions,
            obs_dim=self._STACK_OBS_DIM,  # 80  ← stacking constraint: 80 % 4 == 0  ✓
            obs_stack_size=obs_stack_size,
            yellow_duration=yellow_duration,
            min_green_duration=min_green_duration,
            reward_weights=reward_weights,
            max_steps=max_steps,
            pressure_normalization_capacity=pressure_normalization_capacity,
            insertion_penalty_clip=insertion_penalty_clip,
            obs_queue_normalization=obs_queue_normalization,
            obs_waiting_time_normalization=obs_waiting_time_normalization,
            speed_threshold=_SCENARIO_PARAMS["sunny"]["speed_threshold"],
            phase_change_penalty=_SCENARIO_PARAMS["sunny"]["phase_change_penalty"],
            sumo_seed=sumo_seed,
        )

    # ------------------------------------------------------------------
    # Public property
    # ------------------------------------------------------------------

    @property
    def total_obs_dim(self) -> int:
        """
        Total per-intersection observation dimension consumed by the agent.

        Returns:
            82  (80-D stacked spatial-temporal features + 2-D context tag)
        """
        return self._STACK_OBS_DIM + self.CONTEXT_DIM

    # ------------------------------------------------------------------
    # Domain-randomization override of reset()
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Sample a scenario, configure reward parameters and route file, then
        delegate to ``super().reset()`` which starts SUMO and returns the
        initial 82-D observation.

        The call chain is:
            MixedTrafficEnv.reset()
                → _sample_and_apply_scenario()   # sets self.route_file etc.
                → super().reset()                 # starts SUMO, fills obs queues
                    → self.get_state()            # calls our overridden get_flat_multi_obs()
                        → MixedTrafficEnv.get_flat_multi_obs()  # returns (N, 82)
        """
        self._sample_and_apply_scenario()
        obs = super().reset()
        # Map the blocked edge to TLS indices after SUMO is running.
        # Safe to call for all scenarios; clears the list for non-accident ones.
        self._resolve_accident_tls()
        return obs

    # ------------------------------------------------------------------
    # Internal: scenario sampling
    # ------------------------------------------------------------------

    def _sample_and_apply_scenario(self) -> None:
        """
        Draw a scenario from the configured distribution and apply:
          • speed_threshold          (used by compute_reward → throughput gate)
          • phase_change_penalty     (used by compute_reward → phase change cost)
          • route_file               (used by start_simulation → SUMO -r flag)
        """
        r = np.random.random()
        p_sunny   = self._scenario_probs["sunny"]
        p_rainy   = self._scenario_probs["rainy"]

        if r < p_sunny:
            scenario = "sunny"
        elif r < p_sunny + p_rainy:
            scenario = "rainy"
        else:
            scenario = "accident"

        self.current_scenario = scenario

        # --- Apply reward parameters ---
        params = _SCENARIO_PARAMS[scenario]
        self.speed_threshold      = float(params["speed_threshold"])
        self.phase_change_penalty = float(params["phase_change_penalty"])

        # --- Apply route file and blocked-edge metadata ---
        if scenario == "sunny":
            self.route_file           = str(np.random.choice(_ROUTES_SUNNY))
            self.current_blocked_edge = ""
            self.current_blocked_lane = ""
        elif scenario == "rainy":
            self.route_file           = str(np.random.choice(_ROUTES_RAINY))
            self.current_blocked_edge = ""
            self.current_blocked_lane = ""
        else:  # accident — pre-baked route file + spatially-aware reward shaping
            self.route_file      = str(np.random.choice(_ROUTES_ACCIDENT))
            # Derive blocked-edge/lane from the route filename stem.
            # e.g. "sumo_files/osm.acc1.rou.xml" → key "osm.acc1"
            _basename = self.route_file.replace("\\\\", "/").split("/")[-1]
            _key      = _basename.split(".rou.xml")[0]
            _edge_info = _ACCIDENT_EDGE_MAP.get(_key, ("", ""))
            self.current_blocked_edge = _edge_info[0]
            self.current_blocked_lane = _edge_info[1]

        print(
            f"[MixedEnv] Scenario: {scenario!r:10s} | "
            f"speed_threshold={self.speed_threshold:.1f} | "
            f"phase_change_penalty={self.phase_change_penalty:.1f} | "
            f"route={self.route_file}"
        )

        self._prepare_dynamic_accident_metadata()

    def _prepare_dynamic_accident_metadata(self) -> None:
        """Load accident_veh stop metadata from current route for runtime activation."""
        self._accident_triggered_runtime = False
        self._accident_route_has_metadata = False
        self._accident_activation_step = None
        self._accident_stop_lane = ""
        self._accident_stop_start_pos = 0.0
        self._accident_stop_end_pos = 0.0

        if not self.dynamic_accident_activation:
            return
        if self.current_scenario != "sunny":
            return
        if not self.route_file:
            return

        route_path = Path(self.route_file)
        if not route_path.exists():
            repo_root = Path(__file__).resolve().parent.parent
            route_path = repo_root / self.route_file
        if not route_path.exists():
            return

        try:
            root = ET.parse(route_path).getroot()
            vehicle = root.find(f".//vehicle[@id='{self.accident_vehicle_id}']")
            if vehicle is None:
                return
            stop = vehicle.find("stop")
            if stop is None:
                return

            lane = (stop.attrib.get("lane") or "").strip()
            start_pos = float(stop.attrib.get("startPos", "0"))
            end_pos = float(stop.attrib.get("endPos", "0"))
            if not lane:
                return

            self._accident_stop_lane = lane
            self._accident_stop_start_pos = min(start_pos, end_pos)
            self._accident_stop_end_pos = max(start_pos, end_pos)
            self._accident_route_has_metadata = True
            self.current_blocked_lane = lane
            self.current_blocked_edge = lane.rsplit("_", 1)[0]

            print(
                f"[MixedEnv] Dynamic accident primed: veh={self.accident_vehicle_id!r}, "
                f"lane={self._accident_stop_lane!r}, "
                f"pos=[{self._accident_stop_start_pos:.1f}, {self._accident_stop_end_pos:.1f}]"
            )
        except Exception as exc:
            print(f"[MixedEnv] Dynamic accident metadata parse failed: {exc}")

    def on_simulation_tick(self) -> None:
        """Switch sunny->accident mode exactly when accident_veh stops at configured stop segment."""
        if not self.dynamic_accident_activation:
            return
        if self.current_scenario != "sunny":
            return
        if self._accident_triggered_runtime:
            return
        if not self._accident_route_has_metadata:
            return

        try:
            if self.accident_vehicle_id not in traci.vehicle.getIDList():
                return

            lane_id = traci.vehicle.getLaneID(self.accident_vehicle_id)
            lane_pos = float(traci.vehicle.getLanePosition(self.accident_vehicle_id))
            speed = float(traci.vehicle.getSpeed(self.accident_vehicle_id))
        except Exception:
            return

        on_target_lane = lane_id == self._accident_stop_lane
        within_stop_segment = (
            self._accident_stop_start_pos - 1.0
            <= lane_pos
            <= self._accident_stop_end_pos + 1.0
        )
        is_stopped = speed <= 0.10

        if not (on_target_lane and within_stop_segment and is_stopped):
            return

        self.current_scenario = "accident"
        accident_params = _SCENARIO_PARAMS["accident"]
        self.speed_threshold = float(accident_params["speed_threshold"])
        self.phase_change_penalty = float(accident_params["phase_change_penalty"])
        self.current_blocked_lane = self._accident_stop_lane
        self.current_blocked_edge = self._accident_stop_lane.rsplit("_", 1)[0]
        self._accident_triggered_runtime = True
        self._accident_activation_step = int(self.current_step)
        self._resolve_accident_tls()

        print(
            f"[MixedEnv] Dynamic accident activated at step={self._accident_activation_step}: "
            f"lane={self.current_blocked_lane!r}, edge={self.current_blocked_edge!r}"
        )

    # ------------------------------------------------------------------
    # Internal: post-reset TLS resolution for accident episodes
    # ------------------------------------------------------------------

    def _resolve_accident_tls(self) -> None:
        """
        Identify which controlled-TLS indices directly serve the blocked edge.
        Called once per episode (after super().reset()) unconditionally.
        For accident scenarios, populates ``self._accident_tls_indices``.
        For non-accident scenarios the list is cleared and returns immediately.
        """
        self._accident_tls_indices = []
        if self.current_scenario != "accident" or not self.current_blocked_edge:
            return
        for idx, tls_id in enumerate(self.controlled_tls_list):
            for lane_id in self.controlled_lanes_per_tls.get(tls_id, []):
                # SUMO lane IDs have the form "<edge_id>_<lane_index>"
                edge_of_lane = lane_id.rsplit("_", 1)[0]
                if edge_of_lane == self.current_blocked_edge:
                    self._accident_tls_indices.append(idx)
                    break  # one match per TLS is sufficient
        if self._accident_tls_indices:
            print(
                f"[MixedEnv] Accident TLS idx={self._accident_tls_indices} "
                f"\u2190 blocked edge={self.current_blocked_edge!r} "
                f"lane={self.current_blocked_lane!r}"
            )
        else:
            print(
                f"[MixedEnv] WARNING: no TLS found for blocked edge "
                f"{self.current_blocked_edge!r}; global weights will apply."
            )

    # ------------------------------------------------------------------
    # Spatially-localized reward override (accident episodes only)
    # ------------------------------------------------------------------

    def compute_reward(self) -> np.ndarray:
        """
        Compute per-intersection rewards with scenario-aware, spatially-
        localized shaping for accident episodes.

        * Non-accident scenarios: delegates entirely to the base-class
          implementation so their reward signal is completely unchanged.
        * Accident scenario: applies relaxed weights (lower pressure penalty,
          higher throughput weight, softer insertion clip) **only** to the
          TLS in ``self._accident_tls_indices``.  All other junctions keep
          standard strict weights so the rest of the network stays efficient.
          A deadlock-prevention penalty (-2.0) fires on the accident TLS
          when >80 % of vehicles on the OPEN lanes of the blocked edge are
          halted (spill-over), not on the blocked lane itself which is always
          congested by the parked vehicle and cannot be controlled.
        """
        if self.current_scenario != "accident":
            return super().compute_reward()

        rewards = np.zeros((self.num_agents,), dtype=np.float32)
        total_waiting_time = 0.0

        # ------------------------------------------------------------------
        # Global insertion signal  (identical formula to base class)
        # ------------------------------------------------------------------
        pending_count = 0.0
        try:
            pending = traci.simulation.getPendingVehicles()
            pending_count = (
                float(len(pending)) if hasattr(pending, "__len__") else float(pending)
            )
        except Exception:
            pending_count = 0.0

        per_agent_pending = pending_count / float(max(self.num_agents, 1))
        insertion_signal  = per_agent_pending / self.pressure_normalization_capacity

        # Store for debug / get_info()
        self.last_pending_count    = float(pending_count)
        self.last_insertion_signal = float(insertion_signal)
        self.last_insertion_component = max(
            self.reward_weights["insertion_penalty"] * insertion_signal,
            self.insertion_penalty_clip,
        )

        # ------------------------------------------------------------------
        # Per-intersection reward computation
        # ------------------------------------------------------------------
        for idx, tls_id in enumerate(self.controlled_tls_list):
            is_accident_tls = idx in self._accident_tls_indices

            # ---- Select per-intersection reward parameters ----
            if is_accident_tls:
                # Relax: forgive unavoidable queue on blocked lane, reward
                # every car that still moves, allow aggressive phase switching.
                throughput_w = 1.5
                pressure_w   = -0.1
                insertion_w  = -0.3
                ins_clip     = -5.0
                phase_w      = self.phase_change_penalty   # -0.2
            else:
                # Unaffected junctions: strict standard weights so the rest
                # of the network does not go idle during the incident.
                throughput_w = self.reward_weights["throughput"]
                pressure_w   = self.reward_weights["pressure"]
                insertion_w  = self.reward_weights["insertion_penalty"]
                ins_clip     = self.insertion_penalty_clip
                phase_w      = -1.0   # standard tight phase-change penalty

            # ---- Insertion component (spatially differentiated) ----
            intersection_insertion = max(insertion_w * insertion_signal, ins_clip)

            # ---- Destination Arrival ----
            throughput = self._get_intersection_throughput_count(tls_id)

            # ---- Pressure (incoming - outgoing, clamped >= 0) ----
            incoming_vehicles = 0.0
            for lane_id in self.controlled_lanes_per_tls.get(tls_id, []):
                try:
                    incoming_vehicles += float(
                        traci.lane.getLastStepVehicleNumber(lane_id)
                    )
                    total_waiting_time += float(traci.lane.getWaitingTime(lane_id))
                except Exception:
                    continue

            outgoing_vehicles = 0.0
            for lane_id in self.outgoing_lanes_per_tls.get(tls_id, []):
                try:
                    outgoing_vehicles += float(
                        traci.lane.getLastStepVehicleNumber(lane_id)
                    )
                except Exception:
                    continue

            pressure            = max(incoming_vehicles - outgoing_vehicles, 0.0)
            normalized_pressure = pressure / self.pressure_normalization_capacity

            # ---- Phase-change penalty ----
            phase_change_component = 0.0
            if (
                idx < len(self.phase_changed_this_step)
                and self.phase_changed_this_step[idx] > 0.0
            ):
                phase_change_component = phase_w

            # ---- Base reward ----
            rewards[idx] = (
                throughput_w * throughput
                + pressure_w * normalized_pressure
                + intersection_insertion
                + phase_change_component
            )

            # ---- Deadlock-prevention penalty (accident TLS only) ----
            # Fire only when spill-over reaches the OPEN lanes on the blocked
            # edge — something the agent can actually prevent with better
            # phasing.  Checking the blocked lane itself is always >0.9 halted
            # (parked truck) and produces uncontrollable gradient noise.
            if is_accident_tls and self.current_blocked_lane:
                try:
                    blocked_edge = self.current_blocked_lane.rsplit("_", 1)[0]
                    open_lanes = [
                        l for l in self.controlled_lanes_per_tls.get(tls_id, [])
                        if l.rsplit("_", 1)[0] == blocked_edge
                        and l != self.current_blocked_lane
                    ]
                    if open_lanes:
                        open_halting = sum(
                            float(traci.lane.getLastStepHaltingNumber(l))
                            for l in open_lanes
                        )
                        open_total = sum(
                            float(traci.lane.getLastStepVehicleNumber(l))
                            for l in open_lanes
                        )
                        if open_total > 0 and (open_halting / open_total) > 0.8:
                            rewards[idx] -= 2.0
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Global tracking (mirrors base-class bookkeeping)
        # ------------------------------------------------------------------
        self.total_waiting_time += total_waiting_time
        self.total_vehicles = len(traci.vehicle.getIDList())
        return rewards

    # ------------------------------------------------------------------
    # Internal: context tag
    # ------------------------------------------------------------------

    def _get_context_tag(self) -> np.ndarray:
        """
        Return the 2-D one-hot environment context tag for the current episode.

        Encoding:
            Sunny / Normal  →  [0., 0.]
            Rainy           →  [1., 0.]
            Accident        →  [0., 1.]

        Returns:
            np.ndarray of shape (2,), dtype float32.
        """
        if self.current_scenario == "rainy":
            return np.array([1.0, 0.0], dtype=np.float32)
        if self.current_scenario == "accident":
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)  # sunny

    # ------------------------------------------------------------------
    # Override get_flat_multi_obs() to append the 2-D context tag
    # ------------------------------------------------------------------

    def get_flat_multi_obs(self) -> np.ndarray:
        """
        Build 82-D per-intersection observations.

        Procedure:
          1. Call ``super().get_flat_multi_obs()`` to obtain the 80-D stacked
             observations via the base-class queue logic.
             Shape: ``(N_agents, 80)``
          2. Build the 2-D context tag for the current episode.
             Shape: ``(2,)``
          3. Tile the tag across all agents.
             Shape: ``(N_agents, 2)``
          4. Concatenate along the feature axis.
             Shape: ``(N_agents, 82)``

        The context tag can switch mid-episode when dynamic accident activation
        is enabled and the configured accident vehicle actually stops.

        Returns:
            np.ndarray of shape ``(N_agents, 82)``, dtype float32.
        """
        base_obs = super().get_flat_multi_obs()           # (N_agents, 80)
        context  = self._get_context_tag()                # (2,)
        tiled    = np.tile(context, (self.num_agents, 1)) # (N_agents, 2)
        return np.concatenate([base_obs, tiled], axis=1)  # (N_agents, 82)

    # ------------------------------------------------------------------
    # get_info() extension: surface scenario metadata
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        """
        Extend base-class info with mixed-environment scenario metadata.

        Returns:
            dict — all base-class keys, plus:
                'scenario'           : str  — current scenario name
                'context_tag'        : list — [rain_flag, accident_flag]
                'speed_threshold'    : float
                'phase_change_penalty': float
        """
        info = super().get_info()
        info["scenario"]             = self.current_scenario
        info["context_tag"]          = self._get_context_tag().tolist()
        info["speed_threshold"]      = float(self.speed_threshold)
        info["phase_change_penalty"] = float(self.phase_change_penalty)
        info["blocked_edge"]         = self.current_blocked_edge
        info["blocked_lane"]         = self.current_blocked_lane
        info["accident_tls_indices"] = list(self._accident_tls_indices)
        info["dynamic_accident_enabled"] = bool(self.dynamic_accident_activation)
        info["dynamic_accident_triggered"] = bool(self._accident_triggered_runtime)
        info["dynamic_accident_activation_step"] = self._accident_activation_step
        info["dynamic_accident_vehicle_id"] = self.accident_vehicle_id
        return info

    # ------------------------------------------------------------------
    # Private: input validation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_probs(scenario_probs: Optional[Dict[str, float]]) -> Dict[str, float]:
        """
        Validate and return a normalised probability dict.

        Raises:
            ValueError if keys are missing or values do not sum to 1.
        """
        if scenario_probs is None:
            return dict(_DEFAULT_PROBS)

        for key in _VALID_SCENARIOS:
            if key not in scenario_probs:
                raise ValueError(
                    f"scenario_probs is missing required key '{key}'. "
                    f"Expected all of: {sorted(_VALID_SCENARIOS)}."
                )

        unknown = set(scenario_probs) - _VALID_SCENARIOS
        if unknown:
            raise ValueError(
                f"scenario_probs contains unknown keys: {unknown}. "
                f"Valid keys: {sorted(_VALID_SCENARIOS)}."
            )

        total = sum(scenario_probs.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"scenario_probs values must sum to 1.0; got {total:.8f}."
            )

        return {k: float(v) for k, v in scenario_probs.items()}
