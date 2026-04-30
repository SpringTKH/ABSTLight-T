"""Testing entrypoint for ABSTLight experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import ABSTConfig
from train import Trainer, find_latest_checkpoint
from agent.abst_light_agent import ABSTLightAgent
from environment.traffic_env_mixed import MixedTrafficEnv
from utils.metrics_tracker import MetricsTracker

_ROUTE_RESULTS_SUBDIR_MAP = {
    "osm.moderate.rou.xml": Path("normal"),
    "osm.peak.rou.xml": Path("peak-hours"),
    "osm.rain_m.rou.xml": Path("rainy") / "moderate_rain",
    "osm.rain_h.rou.xml": Path("rainy") / "heavy_rain",
    "osm.acc1.rou.xml": Path("accident") / "acc1",
    "osm.acc2.rou.xml": Path("accident") / "acc2",
    "osm.acc3.rou.xml": Path("accident") / "acc3",
    "osm.acc4.rou.xml": Path("accident") / "acc4",
}



def _extract_episode_number(dirname: str) -> int:
    match = re.match(r"episode_(\d+)$", dirname)
    return int(match.group(1)) if match else -1


def find_latest_trained_model(models_root: str) -> str | None:
    """Return latest valid episode directory, or None if not found."""
    root = Path(models_root)
    if not root.exists() or not root.is_dir():
        return None

    candidates: list[tuple[int, str]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue

        episode_num = _extract_episode_number(path.name)
        if episode_num < 0:
            continue

        has_state = any(path.glob("agent_state_episode_*.pth")) or (path / "agent_state.pth").exists()
        has_network = (
            any(path.glob("embedding*.pth"))
            and any(path.glob("gcn_mha*.pth"))
            and any(path.glob("q_head_agent*_net*.pth"))
        )

        if has_state and has_network:
            candidates.append((episode_num, str(path)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _ensure_sumo_tools() -> None:
    """Ensure SUMO tools path is available for traci/sumolib imports."""
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME is not set. Please set it before running tests.")

    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _episode_result_template() -> dict:
    return MetricsTracker.episode_result_template()


def _collect_step_metrics(controlled_lanes: list[str]) -> dict:
    import traci

    lane_speeds: list[float] = []
    queue_length = 0.0
    lane_vehicle_ids: set[str] = set()

    for lane_id in controlled_lanes:
        try:
            lane_speeds.append(float(traci.lane.getLastStepMeanSpeed(lane_id)))
            queue_length += float(traci.lane.getLastStepHaltingNumber(lane_id))
            lane_vehicle_ids.update(traci.lane.getLastStepVehicleIDs(lane_id))
        except Exception:
            continue

    # Compute delay as normalized speed deficit relative to allowed speed.
    vehicle_delays: dict[str, float] = {}
    moving_count = 0
    waiting_time_samples: list[float] = []
    time_loss_samples: list[float] = []
    stopped_vehicle_ids: set[str] = set()
    for veh_id in lane_vehicle_ids:
        try:
            speed = float(traci.vehicle.getSpeed(veh_id))
            allowed_speed = float(traci.vehicle.getAllowedSpeed(veh_id))
            if speed > 0.1:
                moving_count += 1
            else:
                stopped_vehicle_ids.add(veh_id)
            if allowed_speed > 1e-6:
                vehicle_delays[veh_id] = max(0.0, (allowed_speed - speed) / allowed_speed)
            else:
                vehicle_delays[veh_id] = 0.0

            try:
                waiting_time_samples.append(float(traci.vehicle.getAccumulatedWaitingTime(veh_id)))
            except Exception:
                waiting_time_samples.append(0.0)
            try:
                time_loss_samples.append(float(traci.vehicle.getTimeLoss(veh_id)))
            except Exception:
                time_loss_samples.append(0.0)
        except Exception:
            vehicle_delays[veh_id] = 0.0

    # CO2 emission is in mg/s; with 1s step this sums to mg per step.
    co2_mg = 0.0
    fuel_mg = 0.0
    try:
        for veh_id in traci.vehicle.getIDList():
            co2_mg += float(traci.vehicle.getCO2Emission(veh_id))
            try:
                fuel_mg += float(traci.vehicle.getFuelConsumption(veh_id))
            except Exception:
                pass
    except Exception:
        pass

    teleport_count = 0
    collision_count = 0
    try:
        teleport_count = int(traci.simulation.getEndingTeleportNumber())
    except Exception:
        teleport_count = 0
    try:
        collision_count = int(traci.simulation.getCollidingVehiclesNumber())
    except Exception:
        collision_count = 0

    green_wave_effectiveness = 0.0
    if lane_vehicle_ids:
        green_wave_effectiveness = float(moving_count / len(lane_vehicle_ids))

    # SUMO fuel is reported in mg/s; convert to litres with gasoline density.
    # litres = mg / (1e6 mg/kg * 0.745 kg/L)
    fuel_litre = float(fuel_mg / (1_000_000.0 * 0.745))

    return {
        "avg_lane_speed": _mean(lane_speeds),
        "queue_length": queue_length,
        "vehicle_delays": vehicle_delays,
        "avg_waiting_time": _mean(waiting_time_samples),
        "avg_time_loss": _mean(time_loss_samples),
        "stopped_vehicle_ids": stopped_vehicle_ids,
        "lane_vehicle_ids": lane_vehicle_ids,
        "collision_teleport_count": int(min(teleport_count, collision_count)),
        "fuel_litre": fuel_litre,
        "green_wave_effectiveness": green_wave_effectiveness,
        "co2_mg": co2_mg,
    }


def _finalize_episode_metrics(
    *,
    simulation_seconds: int,
    arrived_total: int,
    step_speed_samples: list[float],
    step_queue_samples: list[float],
    step_waiting_time_samples: list[float],
    step_time_loss_samples: list[float],
    step_green_samples: list[float],
    vehicle_delay_agg: dict[str, list[float]],
    vehicle_seen_ids: set[str],
    vehicle_stop_seconds: dict[str, int],
    teleportation_by_collisions: int,
    fuel_total_litre: float,
    phase_switch_count: int = 0,
    co2_total_mg: float,
    decision_steps: int | None = None,
) -> dict:
    result = _episode_result_template()
    result["simulation_seconds"] = simulation_seconds
    if decision_steps is not None:
        result["decision_steps"] = decision_steps
    result["arrived_vehicles"] = arrived_total
    result["avg_speed_incoming_road"] = _mean(step_speed_samples)
    result["avg_queue_length"] = _mean(step_queue_samples)
    result["max_queue_length"] = float(max(step_queue_samples)) if step_queue_samples else 0.0
    result["queue_length_variance"] = float(np.std(np.asarray(step_queue_samples, dtype=np.float32))) if step_queue_samples else 0.0
    result["avg_waiting_time"] = _mean(step_waiting_time_samples)
    result["time_loss"] = _mean(step_time_loss_samples)
    result["phase_switch_count"] = int(phase_switch_count)
    if vehicle_seen_ids:
        stopped_seconds_total = float(sum(int(vehicle_stop_seconds.get(vehicle_id, 0)) for vehicle_id in vehicle_seen_ids))
        result["avg_stop_time_per_vehicle"] = float(stopped_seconds_total / len(vehicle_seen_ids))
    else:
        result["avg_stop_time_per_vehicle"] = 0.0
    result["teleportation_by_collisions"] = int(teleportation_by_collisions)
    result["fuel_consumption_total_litre"] = float(fuel_total_litre)
    result["green_wave_effectiveness"] = _mean(step_green_samples)
    result["co2_emissions_total_mg"] = co2_total_mg

    per_vehicle_delay_means: list[float] = []
    for delay_samples in vehicle_delay_agg.values():
        if delay_samples:
            per_vehicle_delay_means.append(_mean(delay_samples))
    result["avg_delay_per_vehicle"] = _mean(per_vehicle_delay_means)

    if simulation_seconds > 0:
        throughput_per_second = float(arrived_total / simulation_seconds)
        result["intersection_throughput_veh_per_step"] = throughput_per_second
        result["intersection_throughput_veh_per_hour"] = float(throughput_per_second * 3600.0)

    return result


def _aggregate_test_results(episode_results: list[dict]) -> dict:
    return MetricsTracker.aggregate_episode_results(episode_results)


def _collect_and_accumulate_tick_metrics(
    *,
    controlled_lanes: list[str],
    step_speed_samples: list[float],
    step_queue_samples: list[float],
    step_waiting_time_samples: list[float],
    step_time_loss_samples: list[float],
    step_green_samples: list[float],
    vehicle_delay_agg: dict[str, list[float]],
    vehicle_seen_ids: set[str],
    vehicle_stop_seconds: dict[str, int],
) -> tuple[float, int, int, float]:
    """Collect one SUMO tick and return CO2, arrivals, teleports, and fuel.

    Returns:
        Tuple ``(co2_mg, arrived_count_for_tick, collision_teleport_count, fuel_litre)``.
    """
    import traci

    metrics = _collect_step_metrics(controlled_lanes)
    step_speed_samples.append(metrics["avg_lane_speed"])
    step_queue_samples.append(metrics["queue_length"])
    step_waiting_time_samples.append(metrics["avg_waiting_time"])
    step_time_loss_samples.append(metrics["avg_time_loss"])
    if metrics["vehicle_delays"]:
        step_green_samples.append(metrics["green_wave_effectiveness"])

    for vehicle_id, delay_value in metrics["vehicle_delays"].items():
        vehicle_delay_agg.setdefault(vehicle_id, []).append(delay_value)

    for vehicle_id in metrics["lane_vehicle_ids"]:
        vehicle_seen_ids.add(vehicle_id)
    for vehicle_id in metrics["stopped_vehicle_ids"]:
        vehicle_stop_seconds[vehicle_id] = int(vehicle_stop_seconds.get(vehicle_id, 0)) + 1

    # Per-SUMO-tick arrivals; sum directly to avoid fragile delta logic.
    arrived_count_for_tick = int(traci.simulation.getArrivedNumber())
    return (
        float(metrics["co2_mg"]),
        arrived_count_for_tick,
        int(metrics["collision_teleport_count"]),
        float(metrics["fuel_litre"]),
    )


def _abstlight_step_with_tick_sampling(env, actions: np.ndarray, on_tick) -> tuple[np.ndarray, np.ndarray, bool, dict]:
    """Mirror env.step semantics while exposing each internal SUMO tick via callback."""
    import traci

    requested_actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    if requested_actions.shape[0] != env.num_agents:
        raise ValueError(f"Expected {env.num_agents} actions, got {requested_actions.shape[0]}.")

    phase_switch_count_step = 0

    env.available_phase_count_per_tls = {
        tls_id: env._get_phase_count(tls_id)
        for tls_id in env.controlled_tls_list
    }
    env.available_phase_count = min(env.available_phase_count_per_tls.values())
    env.num_actions = min(env.configured_num_actions, env.available_phase_count)

    safe_actions = requested_actions.copy()
    safe_actions = env._apply_safety_fallback(safe_actions, is_eval=True)
    changed_indices = []
    for idx, tls_id in enumerate(env.controlled_tls_list):
        phase_count = max(1, int(env.available_phase_count_per_tls[tls_id]))
        safe_actions[idx] = int(safe_actions[idx]) % phase_count
        if safe_actions[idx] != int(env.current_phase[idx]):
            changed_indices.append(idx)

    if changed_indices:
        phase_switch_count_step = len(changed_indices)
        for _ in range(env.yellow_duration):
            for idx in changed_indices:
                tls_id = env.controlled_tls_list[idx]
                traci.trafficlight.setPhase(tls_id, 1)
            traci.simulationStep()
            env.current_step += 1
            if hasattr(env, "on_simulation_tick"):
                env.on_simulation_tick()
            on_tick()
            if env.current_step >= env.max_steps:
                break

        for idx in changed_indices:
            tls_id = env.controlled_tls_list[idx]
            try:
                traci.trafficlight.setPhase(tls_id, int(safe_actions[idx]))
                env.current_phase[idx] = int(safe_actions[idx])
            except Exception:
                traci.trafficlight.setPhase(tls_id, 0)
                env.current_phase[idx] = 0
            env.time_since_last_change[idx] = 0

    for _ in range(env.min_green_duration):
        if env.current_step >= env.max_steps:
            break
        traci.simulationStep()
        env.current_step += 1
        env.time_since_last_change += 1
        if hasattr(env, "on_simulation_tick"):
            env.on_simulation_tick()
        on_tick()

    next_state = env.get_state()
    reward = env.compute_reward()
    done = env.current_step >= env.max_steps
    info = env.get_info()
    info["phase_switch_count_step"] = int(phase_switch_count_step)
    return next_state, reward, done, info


def _save_test_results_file(payload: dict) -> str:
    route_file = str(payload.get("route_file", "") or "")
    route_name = Path(route_file).name.lower()

    base_dir = Path("results")
    scenario_subdir = _ROUTE_RESULTS_SUBDIR_MAP.get(route_name, Path("misc"))
    output_dir = base_dir / scenario_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_label = payload.get("model_type", "unknown").lower().replace(" ", "_")
    output_path = output_dir / f"test_results_{model_label}_{timestamp}.json"

    tracker = MetricsTracker(model_name=str(payload.get("model_type", "unknown")), scenario=str(payload.get("scenario", "")))
    tracker.report_payload = payload
    return tracker.save_results_to_json(output_path)


def _list_generated_route_files(routes_dir: str) -> list[str]:
    """Return sorted route XML files from a directory."""
    root = Path(routes_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Routes directory not found: {routes_dir}")

    route_files = sorted(str(path) for path in root.glob("*.rou.xml") if path.is_file())
    if not route_files:
        raise FileNotFoundError(f"No '*.rou.xml' files found in: {routes_dir}")
    return route_files


# ============================================================================
# Model 4 – Fine-Tuned 3-in-1  (82-D architecture, deterministic scenario)
# ============================================================================

class _DeterministicMixedEnv(MixedTrafficEnv):
    """
    MixedTrafficEnv variant for deterministic test evaluation.

    Overrides ``_sample_and_apply_scenario()`` to lock the scenario to a
    single value (sunny / rainy / accident) chosen at construction time,
    instead of drawing from a random distribution every episode.

    The 80-D / 82-D obs_dim split, stacking constraint, and context-tag
    logic are all inherited unchanged from MixedTrafficEnv.

    Args
    ----
    forced_scenario : One of ``'sunny'``, ``'rainy'``, or ``'accident'``.
    forced_route_file : Override the route file for every episode. When
        ``None`` the normal per-scenario random file selection applies.
    All other kwargs are forwarded directly to ``MixedTrafficEnv.__init__``.
    """

    _SCENARIO_PARAMS = {
        "sunny":    {"speed_threshold": 2.0, "phase_change_penalty": -1.0},
        "rainy":    {"speed_threshold": 1.4, "phase_change_penalty": -0.3},
        "accident": {"speed_threshold": 2.0, "phase_change_penalty": -1.0},
    }

    def __init__(
        self,
        *args,
        forced_scenario: str,
        forced_route_file: str | None = None,
        dynamic_accident_activation: bool = False,
        **kwargs,
    ):
        if forced_scenario not in self._SCENARIO_PARAMS:
            raise ValueError(
                f"forced_scenario must be one of {list(self._SCENARIO_PARAMS)}, "
                f"got {forced_scenario!r}"
            )
        self._forced_scenario = forced_scenario
        self._forced_route_file = forced_route_file
        self._dynamic_accident_activation = bool(dynamic_accident_activation)
        # Neutralise the random sampler by making only one scenario reachable.
        kwargs["scenario_probs"] = {
            "sunny":    1.0 if forced_scenario == "sunny"    else 0.0,
            "rainy":    1.0 if forced_scenario == "rainy"    else 0.0,
            "accident": 1.0 if forced_scenario == "accident" else 0.0,
        }
        kwargs["dynamic_accident_activation"] = self._dynamic_accident_activation
        super().__init__(*args, **kwargs)

    def _sample_and_apply_scenario(self) -> None:
        """Lock scenario to the value chosen at construction."""
        super()._sample_and_apply_scenario()
        # After the parent has set route_file via random.choice, override it
        # with the caller-specified file if one was provided.
        if self._forced_route_file is not None:
            self.route_file = self._forced_route_file
            print(f"[DeterministicEnv] Forcing route_file: {self.route_file}")
            # Re-derive blocked-edge/lane metadata from the forced route file
            # so that accident reward shaping targets the correct edge.
            if self._forced_scenario == "accident":
                from environment.traffic_env_mixed import _ACCIDENT_EDGE_MAP
                _basename = self.route_file.replace("\\", "/").split("/")[-1]
                _key = _basename.split(".rou.xml")[0]
                _edge_info = _ACCIDENT_EDGE_MAP.get(_key, ("", ""))
                self.current_blocked_edge = _edge_info[0]
                self.current_blocked_lane = _edge_info[1]


def run_finetuned_mode(
    test_episodes: int,
    max_steps: int,
    use_gui: bool,
    checkpoint_dir: str | None,
    route_file: str | None = None,
    rainy: bool = False,
    accident: bool = False,
    seed: int = 42,
) -> dict:
    """
    Run evaluation of the fine-tuned 3-in-1 ABSTLight model (--model 4).

    Architecture
    ------------
    * 82-D observations: 80-D stacked spatial-temporal + 2-D context tag.
    * Deterministic scenario: fixed to sunny / rainy / accident based on CLI
      flags (no domain randomization during testing).
    * Pure exploitation: epsilon forced to 0.0.

    Checkpoint resolution
    ---------------------
    If ``checkpoint_dir`` is None the function looks for the latest episode
    directory inside ``model_finetune/`` via ``find_latest_checkpoint()``.
    The agent's ``load()`` method handles episodic filenames automatically.

    Context tag assignment
    ----------------------
    --rainy             →  [1, 0]  speed_threshold=1.4  phase_penalty=-0.3
    --accident          →  [0, 1]  speed_threshold=2.0  phase_penalty=-1.0
    (neither)           →  [0, 0]  speed_threshold=2.0  phase_penalty=-1.0

    Args
    ----
    test_episodes    : Number of episodes to evaluate.
    max_steps        : SUMO steps per episode.
    use_gui          : Launch SUMO-GUI if True.
    checkpoint_dir   : Path to fine-tuned checkpoint folder. None → auto-detect.
    route_file       : Override route file passed to SUMO. None → scenario default.
    rainy            : Activate rainy scenario context.
    accident         : Activate accident scenario context.

    Returns
    -------
    dict with keys: ``model_type``, ``mode``, ``checkpoint_dir``,
    ``route_file``, ``scenario``, ``episodes``, ``aggregate``.
    """
    _ensure_sumo_tools()
    import traci

    # ------------------------------------------------------------------
    # 1. Resolve checkpoint directory
    # ------------------------------------------------------------------
    _FINETUNE_MODEL_DIR = "model_finetune"
    if checkpoint_dir is None:
        ckpt_dir, ckpt_episode = find_latest_checkpoint(_FINETUNE_MODEL_DIR)
        if ckpt_dir is None:
            raise FileNotFoundError(
                f"No fine-tuned checkpoint found in '{_FINETUNE_MODEL_DIR}/'. "
                "Run train_finetune.py before evaluating --model 4."
            )
        checkpoint_dir = ckpt_dir
    else:
        ckpt_episode = _extract_episode_number(Path(checkpoint_dir).name)
        if ckpt_episode < 0:
            ckpt_episode = None

    print(f"[FineTuned] Checkpoint : {checkpoint_dir}  (episode={ckpt_episode})")

    # ------------------------------------------------------------------
    # 2. Determine fixed scenario from CLI flags
    # ------------------------------------------------------------------
    if rainy:
        forced_scenario = "rainy"
    elif accident:
        forced_scenario = "accident"
    else:
        forced_scenario = "sunny"

    print(f"[FineTuned] Forced scenario : {forced_scenario!r}")

    route_for_dynamic_check = (route_file or "").replace("\\", "/").lower()
    dynamic_accident_activation = (
        forced_scenario == "sunny"
        and ".acc" in route_for_dynamic_check
    )
    if dynamic_accident_activation:
        print("[FineTuned] Dynamic accident switch enabled (sunny -> accident on accident_veh stop event)")

    # ------------------------------------------------------------------
    # 3. Build deterministic 82-D environment
    #
    #    _DeterministicMixedEnv inherits all 80→82D obs logic from
    #    MixedTrafficEnv; we only lock the scenario (and optionally the
    #    route file) so every episode uses the same context tag.
    # ------------------------------------------------------------------
    env = _DeterministicMixedEnv(
        sumo_config=ABSTConfig.SUMO_CONFIG_PATH,
        use_gui=use_gui,
        num_actions=ABSTConfig.NUM_ACTIONS,
        obs_stack_size=int(getattr(ABSTConfig, "OBS_STACK_SIZE", 4)),
        yellow_duration=ABSTConfig.YELLOW_PHASE_DURATION,
        min_green_duration=ABSTConfig.MIN_GREEN_DURATION,
        max_steps=max_steps,
        pressure_normalization_capacity=float(
            getattr(ABSTConfig, "PRESSURE_NORMALIZATION_CAPACITY", 50.0)
        ),
        insertion_penalty_clip=float(
            getattr(ABSTConfig, "INSERTION_PENALTY_CLIP", -30.0)
        ),
        obs_queue_normalization=float(
            getattr(ABSTConfig, "OBS_QUEUE_NORMALIZATION", 50.0)
        ),
        obs_waiting_time_normalization=float(
            getattr(ABSTConfig, "OBS_WAITING_TIME_NORMALIZATION", 300.0)
        ),
        forced_scenario=forced_scenario,
        forced_route_file=route_file,  # None → scenario's own random choice
        dynamic_accident_activation=dynamic_accident_activation,
        sumo_seed=int(seed),
    )

    num_agents  = env.detect_num_agents()
    num_actions = env.detect_action_space()
    obs_dim     = env.total_obs_dim  # 82

    print(f"[FineTuned] obs_dim={obs_dim}  agents={num_agents}  actions={num_actions}")

    # ------------------------------------------------------------------
    # 4. Build ABSTLightAgent (82-D, pure exploitation)
    # ------------------------------------------------------------------
    adj_matrix = env.get_adjacency_matrix().astype(np.float32)

    agent = ABSTLightAgent(
        obs_dim=obs_dim,
        num_agents=num_agents,
        num_actions=num_actions,
        adj_matrix=adj_matrix,
        embed_dim=ABSTConfig.EMBED_DIM,
        num_heads=ABSTConfig.NUM_HEADS,
        num_gcn_layers=ABSTConfig.NUM_GCN_LAYERS,
        gcn_dropout=ABSTConfig.GCN_DROPOUT,
        hidden_dim=ABSTConfig.Q_HEAD_HIDDEN_DIM,
        N=ABSTConfig.N_NETWORKS,
        M=ABSTConfig.M_SUBSET_SIZE,
        learning_rate=0.0,       # no training steps taken during evaluation
        epsilon_start=0.0,       # pure exploitation — no random actions
        epsilon_end=0.0,
        epsilon_decay=1.0,
        device="cuda" if (getattr(ABSTConfig, "USE_CUDA", False) and torch.cuda.is_available()) else "cpu",
    )

    # ------------------------------------------------------------------
    # 5. Load fine-tuned weights and set eval mode
    # ------------------------------------------------------------------
    if ckpt_episode is not None:
        agent.load(checkpoint_dir, episode=ckpt_episode)
    else:
        agent.load(checkpoint_dir)

    agent.eval_mode()
    agent.epsilon = 0.0  # belt-and-suspenders: bypass set_epsilon clamping
    print(f"[FineTuned] Epsilon forced to 0.0 (pure exploitation)")

    # ------------------------------------------------------------------
    # 6. Evaluation loop  (mirrors run_abstlight_mode structure)
    # ------------------------------------------------------------------
    episode_results: list[dict] = []
    test_rewards: list[float] = []

    for episode in range(1, test_episodes + 1):
        raw_state = env.reset()
        state = np.asarray(raw_state, dtype=np.float32)
        if state.ndim == 1 and num_agents == 1:
            state = state.reshape(1, obs_dim)

        controlled_lanes = list(env.controlled_lanes)
        tls_ids = list(env.controlled_tls_list)
        previous_phase_states = {
            tls_id: int(traci.trafficlight.getPhase(tls_id))
            for tls_id in tls_ids
        }
        tracker = MetricsTracker(
            model_name="ABSTLIGHT_FINETUNED",
            scenario=forced_scenario,
            num_agents=num_agents,
        )

        done = False
        decision_steps = 0
        episode_reward = 0.0
        latest_info: dict = {}

        def on_tick() -> None:
            tracker.update_per_step(controlled_lanes)
            for tls_id in tls_ids:
                try:
                    current_phase = int(traci.trafficlight.getPhase(tls_id))
                except Exception:
                    continue
                if tls_id in previous_phase_states and current_phase != previous_phase_states[tls_id]:
                    tracker.register_phase_switches(1)
                previous_phase_states[tls_id] = current_phase

        while not done:
            state_tensor = torch.from_numpy(state).float()
            with torch.no_grad():
                # Use "averaging" (sum Q-values across all heads, then argmax) instead
                # of "voting" to eliminate np.random.choice tie-breaking stochasticity.
                # With epsilon=0 and averaging, action selection is fully deterministic
                # given the model weights, which is required for stable eval results.
                actions = agent.select_all_actions(state_tensor, mode="averaging")
            actions_np = np.asarray(actions, dtype=np.int64)
            next_raw, reward, done, latest_info = _abstlight_step_with_tick_sampling(
                env, actions_np, on_tick,
            )
            next_state = np.asarray(next_raw, dtype=np.float32)
            if next_state.ndim == 1 and num_agents == 1:
                next_state = next_state.reshape(1, obs_dim)
            state = next_state

            reward_array = np.asarray(reward, dtype=np.float32)
            episode_reward += float(np.sum(reward_array))
            decision_steps += 1

        episode_result = tracker.end_episode(
            total_reward=episode_reward,
            arrived_vehicles=tracker.arrived_total,
            decision_steps=decision_steps,
            safety_fallback_count=int(latest_info.get("safety_fallback_count", 0)),
            extra_fields={"scenario": forced_scenario},
        )
        test_rewards.append(episode_reward)
        episode_results.append(episode_result)

        print(f"\nTest Episode {episode}/{test_episodes}  [{forced_scenario}]")
        print(f"  Reward: {episode_reward:.2f}  |  Avg: {_mean(test_rewards):.2f}")
        print(f"  Decision Steps: {decision_steps}  |  Simulation Seconds: {tracker.simulation_seconds}")
        print(f"  Safety Fallback Triggers: {episode_result['safety_fallback_count']}")

    env.close()

    aggregate = _aggregate_test_results(episode_results)
    aggregate["avg_reward"] = _mean(test_rewards)

    print("\n" + "=" * 60)
    print("Fine-Tuned Model Testing Completed!")
    print("=" * 60)
    print(f"Scenario         : {forced_scenario}")
    print(f"Average Reward   : {aggregate['avg_reward']:.2f}")
    print(f"Avg Delay/Vehicle: {aggregate['avg_delay_per_vehicle']:.4f}")
    print(f"Avg Incoming Speed: {aggregate['avg_speed_incoming_road']:.4f}")
    print(f"Avg Queue Length : {aggregate['avg_queue_length']:.4f}")
    print(f"Max Queue Length : {aggregate['max_queue_length']:.4f}")
    print(f"Queue Std Dev    : {aggregate['queue_length_variance']:.4f}")
    print(f"Avg Waiting Time : {aggregate['avg_waiting_time']:.4f}")
    print(f"Avg Time Loss    : {aggregate['time_loss']:.4f}")
    print(f"Phase Switches   : {aggregate['phase_switch_count']}")
    print(f"Avg Stop Time/Veh: {aggregate['avg_stop_time_per_vehicle']:.4f}")
    print(f"Teleport(Coll.)  : {aggregate['teleportation_by_collisions']}")
    print(f"Fuel Total (L)   : {aggregate['fuel_consumption_total_litre']:.6f}")
    print(
        "Throughput (veh/step | veh/hour): "
        f"{aggregate['intersection_throughput_veh_per_step']:.4f} | "
        f"{aggregate['intersection_throughput_veh_per_hour']:.2f}"
    )
    print(f"Green Wave Eff.  : {aggregate['green_wave_effectiveness']:.4f}")
    print(f"CO2 Total (mg)   : {aggregate['co2_emissions_total_mg']:.2f}")

    return {
        "model_type": "ABSTLIGHT_FINETUNED",
        "mode": 4,
        "seed": int(seed),
        "checkpoint_dir": checkpoint_dir,
        "route_file": route_file,
        "scenario": forced_scenario,
        "episodes": episode_results,
        "aggregate": aggregate,
    }


def _run_single_mode(
    *,
    model: int,
    max_steps: int,
    test_episodes: int,
    use_gui: bool,
    checkpoint_dir: str | None,
    route_file: str | None,
    speed_threshold: float = 2.0,
    weather: str = "normal",
    phase_penalty: float = -1.0,
    seed: int = 42,
) -> dict:
    """Run one selected mode and return standardized payload."""
    if model == 1:
        return run_sumo_default_mode(
            max_steps=max_steps,
            test_episodes=test_episodes,
            use_gui=use_gui,
            mode=1,
            model_type="SUMO_DEFAULT",
            additional_file=None,
            route_file=route_file,
            seed=seed,
        )
    if model == 2:
        return run_abstlight_mode(
            test_episodes=test_episodes,
            max_steps=max_steps,
            use_gui=use_gui,
            checkpoint_dir=checkpoint_dir,
            route_file=route_file,
            speed_threshold=speed_threshold,
            weather=weather,
            phase_penalty=phase_penalty,
            seed=seed,
        )
    if model == 4:
        return run_finetuned_mode(
            test_episodes=test_episodes,
            max_steps=max_steps,
            use_gui=use_gui,
            checkpoint_dir=checkpoint_dir,
            route_file=route_file,
            rainy=(weather == "rainy"),
            seed=seed,
        )
    return run_sumo_default_mode(
        max_steps=max_steps,
        test_episodes=test_episodes,
        use_gui=use_gui,
        mode=3,
        model_type="SUMO_FIXED_TIME",
        additional_file="sumo_files/fixed_tls.add.xml",
        route_file=route_file,
        seed=seed,
    )


def _save_batch_results_file(payload: dict) -> str:
    """Persist consolidated batch run results."""
    output_dir = Path("results") / "batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"test_results_batch_generated_routes_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return str(output_path)


def run_single_model_for_generated_routes(
    *,
    model: int,
    routes_dir: str,
    max_steps: int,
    test_episodes: int,
    use_gui: bool,
    checkpoint_dir: str | None,
    speed_threshold: float = 2.0,
    weather: str = "normal",
    phase_penalty: float = -1.0,
    seed: int = 42,
) -> tuple[dict, str]:
    """Run one selected model for each generated route file sequentially."""
    route_files = _list_generated_route_files(routes_dir)
    batch_results: list[dict] = []
    total_runs = len(route_files)

    print("\n" + "=" * 60)
    print(f"Batch Test Run: Model {model} x Generated Routes")
    print("=" * 60)
    print(f"Routes directory: {routes_dir}")
    print(f"Route files found: {len(route_files)}")
    print(f"Total runs scheduled: {total_runs}")

    for index, route_file in enumerate(route_files, start=1):
        print("\n" + "-" * 60)
        print(f"Route file: {route_file}")
        print("-" * 60)
        print(f"\n[{index}/{total_runs}] Running model={model} for route={route_file}")

        result_payload = _run_single_mode(
            model=model,
            max_steps=max_steps,
            test_episodes=test_episodes,
            use_gui=use_gui,
            checkpoint_dir=checkpoint_dir,
            route_file=route_file,
            speed_threshold=speed_threshold,
            weather=weather,
            phase_penalty=phase_penalty,
            seed=seed,
        )
        single_saved_path = _save_test_results_file(result_payload)
        print(f"Saved single-run result: {single_saved_path}")
        batch_results.append(result_payload)

    batch_payload = {
        "mode": "single_model_generated_routes",
        "model": model,
        "seed": int(seed),
        "routes_dir": routes_dir,
        "route_files": route_files,
        "total_runs": total_runs,
        "results": batch_results,
    }
    batch_saved_path = _save_batch_results_file(batch_payload)
    return batch_payload, batch_saved_path


def run_sumo_default_mode(
    max_steps: int,
    test_episodes: int,
    use_gui: bool,
    *,
    mode: int = 1,
    model_type: str = "SUMO_DEFAULT",
    additional_file: str | None = None,
    route_file: str | None = None,
    seed: int = 42,
) -> dict:
    """Run SUMO with baseline TLS logic under standardized evaluation."""
    _ensure_sumo_tools()
    import sumolib
    import traci

    sumo_binary = sumolib.checkBinary("sumo-gui" if use_gui else "sumo")
    sumo_cmd = [
        sumo_binary,
        "-c",
        ABSTConfig.SUMO_CONFIG_PATH,
        "--no-step-log",
        "True",
        "--no-warnings",
        "True",
        "--seed",
        str(int(seed)),
    ]
    if additional_file:
        sumo_cmd.extend(["-a", additional_file])
    if route_file:
        sumo_cmd.extend(["-r", route_file])

    done_reason = "max_steps_reached"
    episode_results: list[dict] = []

    for episode in range(1, test_episodes + 1):
        tracker = MetricsTracker(model_name=model_type)
        controlled_lanes: list[str] = []
        previous_phase_states: dict[str, int] = {}
        departed_total = 0

        try:
            traci.start(sumo_cmd)
            tls_ids = traci.trafficlight.getIDList()
            tracker.num_agents = max(1, len(tls_ids))
            if tls_ids:
                controlled_lanes = traci.trafficlight.getControlledLanes(tls_ids[0])
                previous_phase_states = {
                    tls_id: int(traci.trafficlight.getPhase(tls_id))
                    for tls_id in tls_ids
                }

            while tracker.simulation_seconds < max_steps:
                traci.simulationStep()
                departed_total += int(traci.simulation.getDepartedNumber())

                if tls_ids:
                    for tls_id in tls_ids:
                        try:
                            current_phase = int(traci.trafficlight.getPhase(tls_id))
                        except Exception:
                            continue
                        if tls_id in previous_phase_states and current_phase != previous_phase_states[tls_id]:
                            tracker.register_phase_switches(1)
                        previous_phase_states[tls_id] = current_phase

                tracker.update_per_step(controlled_lanes)

            episode_result = tracker.end_episode(
                total_reward=0.0,
                arrived_vehicles=tracker.arrived_total,
                include_reward=False,
            )
            episode_results.append(episode_result)

            print(f"\nEpisode {episode}/{test_episodes}")
            print(f"  Simulation seconds: {tracker.simulation_seconds}")
            print(f"  Stop reason: {done_reason}")
            print(f"  Arrived vehicles: {tracker.arrived_total}")
            print(f"  Departed vehicles: {departed_total}")
            print("  Metrics:")
            print(f"    Avg delay/vehicle: {episode_result['avg_delay_per_vehicle']:.4f}")
            print(f"    Avg incoming speed: {episode_result['avg_speed_incoming_road']:.4f}")
            print(f"    Avg queue length: {episode_result['avg_queue_length']:.4f}")
            print(f"    Max queue length: {episode_result['max_queue_length']:.4f}")
            print(f"    Queue std dev: {episode_result['queue_length_variance']:.4f}")
            print(f"    Avg waiting time: {episode_result['avg_waiting_time']:.4f}")
            print(f"    Avg time loss: {episode_result['time_loss']:.4f}")
            print(f"    Phase switches: {episode_result['phase_switch_count']}")
            print(f"    Avg stop time/vehicle: {episode_result['avg_stop_time_per_vehicle']:.4f}")
            print(f"    Teleportation by collisions: {episode_result['teleportation_by_collisions']}")
            print(f"    Fuel total (litre): {episode_result['fuel_consumption_total_litre']:.6f}")
            print(
                "    Throughput (veh/step | veh/hour): "
                f"{episode_result['intersection_throughput_veh_per_step']:.4f} | "
                f"{episode_result['intersection_throughput_veh_per_hour']:.2f}"
            )
            print(f"    Green wave effectiveness: {episode_result['green_wave_effectiveness']:.4f}")
            print(f"    CO2 total (mg): {episode_result['co2_emissions_total_mg']:.2f}")
        finally:
            try:
                traci.close()
            except Exception:
                pass

    aggregate = _aggregate_test_results(episode_results)

    print("\n" + "=" * 60)
    print(f"{model_type} Mode Completed")
    print("=" * 60)
    print(f"Episodes evaluated: {test_episodes}")
    print(f"Avg delay/vehicle: {aggregate['avg_delay_per_vehicle']:.4f}")
    print(f"Avg incoming speed: {aggregate['avg_speed_incoming_road']:.4f}")
    print(f"Avg queue length: {aggregate['avg_queue_length']:.4f}")

    return {
        "model_type": model_type,
        "mode": mode,
        "seed": int(seed),
        "route_file": route_file,
        "stop_reason": done_reason,
        "episodes": episode_results,
        "aggregate": aggregate,
    }


def run_abstlight_mode(
    test_episodes: int,
    max_steps: int,
    use_gui: bool,
    checkpoint_dir: str | None,
    route_file: str | None = None,
    speed_threshold: float = 2.0,
    weather: str = "normal",
    phase_penalty: float = -1.0,
    seed: int = 42,
) -> dict:
    """Run ABSTLight evaluation with trained->untrained fallback."""

    if checkpoint_dir:
        print(f"[abstlight] Using trained model: {checkpoint_dir}")
    else:
        print("[abstlight] No trained model found, using current untrained agent state.")

    trainer = Trainer(ABSTConfig, use_gui=use_gui, num_episodes=None, route_file=route_file)
    trainer.env.sumo_seed = int(seed)
    trainer.env.max_steps = int(max_steps)
    trainer.env.speed_threshold = float(speed_threshold)
    trainer.env.phase_change_penalty = float(phase_penalty)

    if checkpoint_dir and os.path.exists(checkpoint_dir):
        # Support checkpoint folders named like model/episode_910 where files are
        # saved as *_episode_910.pth.
        episode_from_dir = _extract_episode_number(Path(checkpoint_dir).name)
        if episode_from_dir >= 0:
            trainer.agent.load(checkpoint_dir, episode=episode_from_dir)
        else:
            trainer.agent.load(checkpoint_dir)

    trainer.agent.eval_mode()
    trainer.agent.set_epsilon(0.0)
    # In training, epsilon is clamped by epsilon_end. For test-only behavior,
    # force exact greedy policy with epsilon=0.
    trainer.agent.epsilon = 0.0
    print("[abstlight] Forced epsilon: 0.0 (test-only)")

    import traci

    episode_results: list[dict] = []
    test_rewards: list[float] = []

    for episode in range(1, test_episodes + 1):
        state = trainer._to_flat_multi_agent_obs(trainer.env.reset())
        controlled_lanes = list(trainer.env.controlled_lanes)
        tls_ids = list(trainer.env.controlled_tls_list)
        previous_phase_states = {
            tls_id: int(traci.trafficlight.getPhase(tls_id))
            for tls_id in tls_ids
        }
        tracker = MetricsTracker(
            model_name="ABSTLIGHT",
            num_agents=getattr(trainer.env, "num_agents", 1),
        )

        done = False
        decision_steps = 0
        episode_reward = 0.0
        latest_info: dict = {}

        def on_tick() -> None:
            tracker.update_per_step(controlled_lanes)
            for tls_id in tls_ids:
                try:
                    current_phase = int(traci.trafficlight.getPhase(tls_id))
                except Exception:
                    continue
                if tls_id in previous_phase_states and current_phase != previous_phase_states[tls_id]:
                    tracker.register_phase_switches(1)
                previous_phase_states[tls_id] = current_phase

        while not done:
            state_tensor = torch.from_numpy(state).float()
            actions = trainer.agent.select_all_actions(state_tensor, mode="voting")
            actions_np = np.asarray(actions, dtype=np.int64)
            next_state, reward, done, latest_info = _abstlight_step_with_tick_sampling(
                trainer.env,
                actions_np,
                on_tick,
            )
            state = trainer._to_flat_multi_agent_obs(next_state)

            reward_array = np.asarray(reward, dtype=np.float32)
            episode_reward += float(np.sum(reward_array))
            decision_steps += 1

        episode_result = tracker.end_episode(
            total_reward=episode_reward,
            arrived_vehicles=tracker.arrived_total,
            decision_steps=decision_steps,
            safety_fallback_count=int(latest_info.get("safety_fallback_count", 0)),
        )
        test_rewards.append(episode_reward)
        episode_results.append(episode_result)

        print(f"\nTest Episode {episode}/{test_episodes}")
        print(f"  Reward: {episode_reward:.2f}  |  Avg: {_mean(test_rewards):.2f}")
        print(f"  Decision Steps: {decision_steps}  |  Simulation Seconds: {tracker.simulation_seconds}")
        print(f"  Safety Fallback Triggers: {episode_result['safety_fallback_count']}")

    trainer.env.close()

    aggregate = _aggregate_test_results(episode_results)
    aggregate["avg_reward"] = _mean(test_rewards)

    print("\n" + "=" * 60)
    print("Testing Completed!")
    print("=" * 60)
    print(f"Average Reward: {aggregate['avg_reward']:.2f}")
    print(f"Average Delay per Vehicle: {aggregate['avg_delay_per_vehicle']:.4f}")
    print(f"Average Incoming Speed: {aggregate['avg_speed_incoming_road']:.4f}")
    print(f"Average Queue Length: {aggregate['avg_queue_length']:.4f}")
    print(f"Max Queue Length: {aggregate['max_queue_length']:.4f}")
    print(f"Queue Std Dev: {aggregate['queue_length_variance']:.4f}")
    print(f"Average Waiting Time: {aggregate['avg_waiting_time']:.4f}")
    print(f"Average Time Loss: {aggregate['time_loss']:.4f}")
    print(f"Phase Switch Count: {aggregate['phase_switch_count']}")
    print(f"Average Stop Time per Vehicle: {aggregate['avg_stop_time_per_vehicle']:.4f}")
    print(f"Teleportation by Collisions: {aggregate['teleportation_by_collisions']}")
    print(f"Fuel Total (litre): {aggregate['fuel_consumption_total_litre']:.6f}")
    print(
        "Throughput (veh/step | veh/hour): "
        f"{aggregate['intersection_throughput_veh_per_step']:.4f} | "
        f"{aggregate['intersection_throughput_veh_per_hour']:.2f}"
    )
    print(f"Green Wave Effectiveness: {aggregate['green_wave_effectiveness']:.4f}")
    print(f"CO2 Total (mg): {aggregate['co2_emissions_total_mg']:.2f}")
    print(f"Total Safety Fallback Triggers: {aggregate['safety_fallback_count']}")

    return {
        "model_type": "ABSTLIGHT",
        "mode": 2,
        "seed": int(seed),
        "checkpoint_dir": checkpoint_dir,
        "route_file": route_file,
        "episodes": episode_results,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ABSTLight test runner")
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help=(
            "1: SUMO default TLS logic, 2: ABSTLight mode, 3: SUMO fixed-time TLS baseline, "
            "4: Fine-tuned 3-in-1 ABSTLight (82-D, loads from model_finetune/)"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=ABSTConfig.MAX_STEPS_PER_EPISODE)
    parser.add_argument("--test-episodes", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--route-file", type=str, default=None)
    parser.add_argument("--routes-dir", type=str, default="sumo_files/generated_routes")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="SUMO random seed used for every simulation activation.")
    parser.add_argument(
        "--accident",
        action="store_true",
        help="Use accident scenario context/reward settings (route file controls accident timing/placement).",
    )
    parser.add_argument(
        "--rainy",
        action="store_true",
        help="MoE gateway: loads model_rain/episode_1000 and applies rainy reward parameters.",
    )
    args = parser.parse_args()

    # Enforce single-episode evaluation across all models.
    args.test_episodes = 1

    weather = "rainy" if args.rainy else "normal"

    # ---------------------------------------------------------------------------
    # MoE Hot-Swap: resolve checkpoint directory from weather context.
    # Only applies to ABSTLight mode (--model 2) when no explicit --checkpoint-dir
    # was provided. Models 1 and 3 are SUMO baselines and need no checkpoint.
    # ---------------------------------------------------------------------------
    if args.model == 2 and args.checkpoint_dir is None:
        if args.rainy:
            args.checkpoint_dir = "model_rain/episode_1000"
        else:
            args.checkpoint_dir = "model/episode_1000"
        print(f"[MoE Gateway] Weather: {weather!r} -> checkpoint: {args.checkpoint_dir}")

    # Resolve speed threshold from weather context.
    # Rainy conditions (tau=1.8, reduced accel) produce lower vehicle speeds;
    # a reduced gate prevents sparse rewards during evaluation.
    speed_threshold = 1.4 if args.rainy else 2.0
    print(f"[MoE Gateway] Weather: {weather!r} -> speed_threshold: {speed_threshold} m/s")

    # Rainy acceleration physics make the standard -1.0 phase-change penalty
    # prohibitive, causing agents to over-hold green lights.
    phase_penalty = -0.3 if args.rainy else -1.0
    print(f"[MoE Gateway] Weather: {weather!r} -> phase_change_penalty: {phase_penalty}")

    route_all_requested = isinstance(args.route_file, str) and args.route_file.strip().lower() == "all"

    # ---------------------------------------------------------------------------
    # Model 4 is a direct dispatch: it manages its own checkpoint, scenario, and
    # observations internally and does not participate in batch/route-all flows.
    # ---------------------------------------------------------------------------
    if args.model == 4:
        results_payload = run_finetuned_mode(
            test_episodes=args.test_episodes,
            max_steps=args.max_steps,
            use_gui=args.gui,
            checkpoint_dir=args.checkpoint_dir,  # None → auto-detect model_finetune/
            route_file=args.route_file if not route_all_requested else None,
            rainy=args.rainy,
            accident=args.accident,
            seed=args.seed,
        )
        saved_path = _save_test_results_file(results_payload)
        print(f"\nSaved test results to: {saved_path}")
        return

    if route_all_requested:
        _, batch_saved_path = run_single_model_for_generated_routes(
            model=args.model,
            routes_dir=args.routes_dir,
            max_steps=args.max_steps,
            test_episodes=args.test_episodes,
            use_gui=args.gui,
            checkpoint_dir=args.checkpoint_dir,
            speed_threshold=speed_threshold,
            weather=weather,
            phase_penalty=phase_penalty,
            seed=args.seed,
        )
        print(f"\nSaved batch test results to: {batch_saved_path}")
    else:
        results_payload = _run_single_mode(
            model=args.model,
            max_steps=args.max_steps,
            test_episodes=args.test_episodes,
            use_gui=args.gui,
            checkpoint_dir=args.checkpoint_dir,
            route_file=args.route_file,
            speed_threshold=speed_threshold,
            weather=weather,
            phase_penalty=phase_penalty,
            seed=args.seed,
        )
        saved_path = _save_test_results_file(results_payload)
        print(f"\nSaved test results to: {saved_path}")


if __name__ == "__main__":
    main()