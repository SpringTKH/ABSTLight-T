"""
Traffic Environment - SUMO/TraCI Wrapper

This module provides a wrapper interface for traffic simulation using
SUMO (Simulation of Urban MObility) via the TraCI (Traffic Control Interface) API.

The environment follows the standard RL interface:
- reset(): Initialize a new episode
- step(action): Execute an action and return (state, reward, done, info)
- get_state(): Get current state representation
- close(): Clean up simulation

State Representation:
- Can be position/velocity matrices, occupancy grids, or visual representations
- Preprocessed into CNN-compatible format (C, H, W)

Actions:
- Traffic signal phase selection
- Timing adjustments

Rewards:
- Based on traffic metrics: waiting time, queue length, throughput, etc.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional, List
import collections
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from config import ABSTConfig

# Check if SUMO is available
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please declare environment variable 'SUMO_HOME'")

try:
    import traci
    import sumolib
except ImportError:
    raise ImportError("TraCI not found. Please install SUMO and set SUMO_HOME.")


class TrafficEnvironment:
    """
    Traffic simulation environment using SUMO/TraCI.
    
    This class provides a gym-like interface for traffic signal control
    using SUMO as the underlying simulator.
    
    Args:
        sumo_config (str): Path to SUMO configuration file (.sumocfg)
        use_gui (bool): Whether to use SUMO GUI or command-line
        num_actions (int): Number of traffic signal phases/actions
        state_shape (tuple): Desired state shape (C, H, W)
        yellow_duration (int): Duration of yellow phase (seconds)
        min_green_duration (int): Minimum green phase duration (seconds)
        reward_weights (dict): Weights for reward function components
        max_steps (int): Maximum simulation steps per episode
    """
    
    def __init__(
        self,
        sumo_config: str,
        use_gui: bool = False,
        route_file: Optional[str] = None,
        num_actions: int = 4,
        state_shape: Tuple[int, int, int] = (1, 84, 84),
        obs_dim: int = None,
        obs_stack_size: int = 1,
        yellow_duration: int = 3,
        min_green_duration: int = 5,
        reward_weights: Dict[str, float] = None,
        max_steps: int = 3600,
        pressure_normalization_capacity: float = 50.0,
        insertion_penalty_clip: float = -5.0,
        obs_queue_normalization: float = 50.0,
        obs_waiting_time_normalization: float = 300.0,
        speed_threshold: float = 2.0,
        phase_change_penalty: float = -1.0,
        sumo_seed: Optional[int] = None,
    ):
        """
        Initialize the traffic environment.
        
        Args:
            sumo_config: Path to .sumocfg file
            use_gui: Use graphical interface
            num_actions: Number of signal phases
            state_shape: State tensor shape
            obs_stack_size: Number of consecutive frames to stack per agent
            yellow_duration: Yellow light duration
            min_green_duration: Minimum green duration
            reward_weights: Reward function weights
            max_steps: Max steps per episode
            pressure_normalization_capacity: Divider for pressure normalization in reward
            insertion_penalty_clip: Maximum penalty per step for insertion delay
            obs_queue_normalization: Divider for queue_length in get_flat_obs
            obs_waiting_time_normalization: Divider for waiting_time in get_flat_obs
        """
        self.sumo_config = sumo_config
        self.use_gui = use_gui
        self.route_file = route_file
        self.configured_num_actions = num_actions
        self.num_actions = num_actions
        self.available_phase_count = num_actions
        self.state_shape = state_shape
        self.obs_dim = int(obs_dim) if obs_dim is not None else None
        self.use_flat_obs = self.obs_dim is not None
        self._obs_stack_size = max(1, int(obs_stack_size))

        if self.use_flat_obs:
            if self.obs_dim % self._obs_stack_size != 0:
                raise ValueError(
                    f"obs_dim ({self.obs_dim}) must be divisible by obs_stack_size ({self._obs_stack_size})."
                )
            self._single_obs_dim = int(self.obs_dim // self._obs_stack_size)
        else:
            self._single_obs_dim = 0

        self.yellow_duration = yellow_duration
        self.min_green_duration = min_green_duration
        self.max_steps = max_steps
        
        # Default reward weights
        if reward_weights is None:
            self.reward_weights = {
                'throughput': 1.0,
                'pressure': -0.5,
                'insertion_penalty': -1.0,
                'phase_change': -0.1
            }
        else:
            # Merge with defaults so callers can override only specific terms.
            self.reward_weights = {
                'throughput': 1.0,
                'pressure': -0.5,
                'insertion_penalty': -1.0,
                'phase_change': -0.1,
            }
            self.reward_weights.update(reward_weights)

        # Reward normalization/clipping controls.
        self.pressure_normalization_capacity = float(pressure_normalization_capacity)
        self.insertion_penalty_clip = float(insertion_penalty_clip)
        self.obs_queue_normalization = float(obs_queue_normalization)
        self.obs_waiting_time_normalization = float(obs_waiting_time_normalization)
        self.speed_threshold = float(speed_threshold)
        self.phase_change_penalty = float(phase_change_penalty)
        self.sumo_seed: Optional[int] = int(sumo_seed) if sumo_seed is not None else None
        
        # SUMO binary
        if use_gui:
            self.sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self.sumo_binary = sumolib.checkBinary('sumo')
        
        # State tracking
        self.current_step = 0
        self.current_phase = 0
        self.time_since_last_change = 0
        self.total_waiting_time = 0
        self.total_vehicles = 0
        self.episode_teleports = 0
        
        # Traffic light IDs (will be populated on reset)
        self.traffic_light_ids = []
        self.controlled_tls = None
        self.controlled_lanes = []
        self.controlled_tls_list = []
        self.controlled_lanes_per_tls: Dict[str, list] = {}
        self.outgoing_lanes_per_tls: Dict[str, list] = {}
        self.induction_loops_per_tls: Dict[str, list] = {}
        self.available_phase_count_per_tls: Dict[str, int] = {}
        self.current_phase = np.zeros((1,), dtype=np.int64)
        self.time_since_last_change = np.zeros((1,), dtype=np.int64)
        self.phase_changed_this_step = np.zeros((1,), dtype=np.float32)
        self.num_agents = 1

        # Safety fallback tracking (evaluation only)
        self._phase_hold_duration = np.zeros((self.num_agents,), dtype=np.float32)
        self._last_phase: List[int] = [0 for _ in range(self.num_agents)]
        self._fallback_triggered_count = 0

        # Observation stack (frame stacking for temporal component)
        # Shape per agent: deque of T raw observations, each (obs_dim_single,)
        self._obs_queues: List[collections.deque] = [
            collections.deque(maxlen=self._obs_stack_size)
            for _ in range(self.num_agents)
        ]

        # Reward debug trackers (updated in compute_reward, exported via get_info).
        self.last_insertion_component = 0.0
        self.last_insertion_signal = 0.0
        self.last_pending_count = 0.0
        
        # Connection status
        self.is_connected = False

        # Per-episode seed counter: incremented on every reset() so that each
        # episode gets a unique SUMO seed, preventing identical episode repeats
        # when the route file is fully deterministic (pre-scheduled vehicles).
        self._episode_count: int = 0
    
    def start_simulation(self):
        """
        Start SUMO simulation with TraCI.
        """
        # Derive a unique seed for this episode from a fixed base + episode counter.
        # Base 23423 is SUMO's built-in default; adding the counter offsets it
        # so successive resets each explore a different stochastic vehicle stream.
        if self.sumo_seed is not None:
            _sumo_seed = int(self.sumo_seed)
        else:
            _sumo_seed = 23423 + self._episode_count
        sumo_cmd = [
            self.sumo_binary,
            '-c', self.sumo_config,
            '--waiting-time-memory', '1000',
            '--time-to-teleport', '-1',
            '--no-step-log', 'True',
            '--no-warnings', 'True',
            '--seed', str(_sumo_seed),
        ]

        if self.route_file:
            sumo_cmd.extend(['-r', self.route_file])
        
        traci.start(sumo_cmd)
        self.is_connected = True
        
        # Get traffic light information
        self.traffic_light_ids = list(traci.trafficlight.getIDList())
        
        if len(self.traffic_light_ids) == 0:
            raise ValueError("No traffic lights found in the simulation!")
        
        # Control all discovered traffic lights for true multi-intersection ABSTLight.
        self.controlled_tls_list = sorted(self.traffic_light_ids)
        self.num_agents = len(self.controlled_tls_list)

        self.controlled_lanes_per_tls = {
            tls_id: list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
            for tls_id in self.controlled_tls_list
        }

        self.outgoing_lanes_per_tls = {}
        for tls_id in self.controlled_tls_list:
            outgoing = []
            links = traci.trafficlight.getControlledLinks(tls_id)
            for signal_links in links:
                for link in signal_links:
                    if not link or len(link) < 2:
                        continue
                    out_lane = link[1]
                    if out_lane:
                        outgoing.append(out_lane)
            self.outgoing_lanes_per_tls[tls_id] = list(dict.fromkeys(outgoing))

        loop_ids = list(traci.inductionloop.getIDList())
        loops_by_lane = {}
        for loop_id in loop_ids:
            try:
                lane_id = traci.inductionloop.getLaneID(loop_id)
            except Exception:
                continue
            loops_by_lane.setdefault(lane_id, []).append(loop_id)

        self.induction_loops_per_tls = {}
        for tls_id in self.controlled_tls_list:
            tls_loops = []
            for lane_id in self.controlled_lanes_per_tls.get(tls_id, []):
                tls_loops.extend(loops_by_lane.get(lane_id, []))
            self.induction_loops_per_tls[tls_id] = list(dict.fromkeys(tls_loops))

        self.controlled_tls = self.controlled_tls_list[0]
        self.controlled_lanes = list(self.controlled_lanes_per_tls[self.controlled_tls])

        self.available_phase_count_per_tls = {
            tls_id: self._get_phase_count(tls_id)
            for tls_id in self.controlled_tls_list
        }
        self.available_phase_count = min(self.available_phase_count_per_tls.values())
        self.num_actions = min(self.configured_num_actions, self.available_phase_count)

        self.current_phase = np.zeros((self.num_agents,), dtype=np.int64)
        self.time_since_last_change = np.zeros((self.num_agents,), dtype=np.int64)
        self.phase_changed_this_step = np.zeros((self.num_agents,), dtype=np.float32)
        self._obs_queues = [
            collections.deque(maxlen=self._obs_stack_size)
            for _ in range(self.num_agents)
        ]
        self._phase_hold_duration = np.zeros((self.num_agents,), dtype=np.float32)
        self._last_phase = [
            int(traci.trafficlight.getPhase(tls_id))
            for tls_id in self.controlled_tls_list
        ]
        self._fallback_triggered_count = 0
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial state observation
        """
        # Close existing connection if any
        if self.is_connected:
            traci.close()
            self.is_connected = False

        # Advance seed counter before (re)starting so each episode uses a
        # distinct SUMO seed for statistically independent traffic realizations.
        self._episode_count += 1

        # Start new simulation
        self.start_simulation()
        
        # Reset counters
        self.current_step = 0
        self.current_phase = np.zeros((self.num_agents,), dtype=np.int64)
        self.time_since_last_change = np.zeros((self.num_agents,), dtype=np.int64)
        self.phase_changed_this_step = np.zeros((self.num_agents,), dtype=np.float32)
        self.total_waiting_time = 0
        self.total_vehicles = 0
        self.last_insertion_component = 0.0
        self.last_insertion_signal = 0.0
        self.last_pending_count = 0.0
        self.episode_teleports = 0
        self._phase_hold_duration = np.zeros((self.num_agents,), dtype=np.float32)
        self._last_phase = [
            int(traci.trafficlight.getPhase(tls_id))
            for tls_id in self.controlled_tls_list
        ]
        self._fallback_triggered_count = 0

        if self.num_actions <= 0:
            self.num_actions = 1

        self.current_phase = np.minimum(self.current_phase, self.num_actions - 1)

        # Set initial traffic light phase for every controlled TLS.
        for idx, tls_id in enumerate(self.controlled_tls_list):
            traci.trafficlight.setPhase(tls_id, int(self.current_phase[idx]))
        
        # Run a few steps to stabilize
        for _ in range(5):
            traci.simulationStep()

        if self.use_flat_obs:
            first_observations = [
                self.get_flat_obs(tls_id)
                for tls_id in self.controlled_tls_list
            ]
            # Pre-fill observation queues with the first observation
            # (Mnih et al. 2015 convention)
            for i, raw_obs in enumerate(first_observations):
                self._obs_queues[i].clear()
                for _ in range(self._obs_stack_size):
                    self._obs_queues[i].append(raw_obs.copy())
        
        # Get initial state
        state = self.get_state()
        
        return state
    
    def step(self, action, is_eval: bool = False) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Traffic signal phase to apply
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Normalize actions to one per controlled intersection.
        if np.isscalar(action):
            requested_actions = np.full((self.num_agents,), int(action), dtype=np.int64)
        else:
            requested_actions = np.asarray(action, dtype=np.int64).reshape(-1)
            if requested_actions.shape[0] != self.num_agents:
                raise ValueError(
                    f"Expected {self.num_agents} actions, got {requested_actions.shape[0]}."
                )

        # Refresh per-TLS phase ranges and clip action space conservatively.
        self.available_phase_count_per_tls = {
            tls_id: self._get_phase_count(tls_id)
            for tls_id in self.controlled_tls_list
        }
        self.available_phase_count = min(self.available_phase_count_per_tls.values())
        self.num_actions = min(self.configured_num_actions, self.available_phase_count)

        safe_actions = requested_actions.copy()
        safe_actions = self._apply_safety_fallback(safe_actions, is_eval=is_eval)
        changed_indices = []
        for idx, tls_id in enumerate(self.controlled_tls_list):
            phase_count = max(1, int(self.available_phase_count_per_tls[tls_id]))
            safe_actions[idx] = int(safe_actions[idx]) % phase_count
            if safe_actions[idx] != int(self.current_phase[idx]):
                changed_indices.append(idx)

        self.phase_changed_this_step = np.zeros((self.num_agents,), dtype=np.float32)
        if changed_indices:
            self.phase_changed_this_step[np.asarray(changed_indices, dtype=np.int64)] = 1.0

        # If at least one TLS changes phase, apply a network-wide yellow transition once.
        if changed_indices:
            for _ in range(self.yellow_duration):
                traci.simulationStep()
                self.current_step += 1
                self.episode_teleports += traci.simulation.getEndingTeleportNumber()
                self._on_post_simulation_step()

            for idx in changed_indices:
                tls_id = self.controlled_tls_list[idx]
                try:
                    traci.trafficlight.setPhase(tls_id, int(safe_actions[idx]))
                    self.current_phase[idx] = int(safe_actions[idx])
                except Exception:
                    traci.trafficlight.setPhase(tls_id, 0)
                    self.current_phase[idx] = 0
                self.time_since_last_change[idx] = 0
        
        # Execute action for minimum green duration
        for _ in range(self.min_green_duration):
            if self.current_step >= self.max_steps:
                break
            
            traci.simulationStep()
            self.current_step += 1
            self.time_since_last_change += 1
            self.episode_teleports += traci.simulation.getEndingTeleportNumber()
            self._on_post_simulation_step()
        
        # Get new state
        next_state = self.get_state()
        
        # Calculate reward
        reward = self.compute_reward()
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps) or (traci.simulation.getMinExpectedNumber() <= 0)
        
        # Additional info
        info = self.get_info()
        
        return next_state, reward, done, info

    def _on_post_simulation_step(self) -> None:
        """
        Optional per-tick hook for subclasses after each traci.simulationStep().

        MixedTrafficEnv uses this to run runtime scenario transitions
        (e.g. sunny -> accident when accident_veh reaches its stop).
        """
        callback = getattr(self, "on_simulation_tick", None)
        if not callable(callback):
            return
        try:
            callback()
        except Exception:
            # Never let optional hooks break the core environment step loop.
            pass

    def _apply_safety_fallback(self, actions: np.ndarray, is_eval: bool = False) -> np.ndarray:
        """
        Enforce an evaluation-only minimum connectivity guarantee.

        If a TLS holds the same phase for too long and the selected action would
        keep holding, override the action to force a phase advance.

        Args:
            actions: Requested phase actions, shape (num_agents,)
            is_eval: Whether current step is evaluation mode

        Returns:
            Possibly modified actions array
        """
        if not is_eval:
            return actions

        max_phase_hold_seconds = int(getattr(ABSTConfig, 'SAFETY_FALLBACK_MAX_PHASE_HOLD', 60))
        modified_actions = np.asarray(actions, dtype=np.int64).copy()

        for i, tls_id in enumerate(self.controlled_tls_list):
            current_phase = int(traci.trafficlight.getPhase(tls_id))
            previous_phase = int(self._last_phase[i]) if i < len(self._last_phase) else current_phase

            if current_phase == previous_phase:
                # Decision cadence follows min green; include yellow when previous
                # decision changed phase to better approximate true hold time.
                step_seconds = int(self.min_green_duration)
                if i < len(self.phase_changed_this_step) and self.phase_changed_this_step[i] > 0.0:
                    step_seconds += int(self.yellow_duration)
                self._phase_hold_duration[i] += float(step_seconds)
            else:
                self._phase_hold_duration[i] = 0.0

            self._last_phase[i] = current_phase

            num_actions = max(1, int(self.available_phase_count_per_tls.get(tls_id, self.num_actions)))
            requested_action = int(modified_actions[i]) % num_actions
            hold_requested = requested_action == current_phase

            if self._phase_hold_duration[i] >= float(max_phase_hold_seconds) and hold_requested:
                forced_action = int((current_phase + 1) % num_actions)
                modified_actions[i] = forced_action
                self._phase_hold_duration[i] = 0.0
                self._fallback_triggered_count += 1
                print(
                    f"[SafetyFallback] TLS {tls_id} held phase {current_phase} "
                    f"for >={max_phase_hold_seconds}s — forcing advance"
                )

        return modified_actions
    

    def get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        This is a simplified state representation. You should customize this
        based on your specific requirements. Options include:
        - Position/velocity matrix
        - Occupancy grid
        - Queue lengths
        - Waiting times
        - Image-based representation
        
        Returns:
            State array of shape specified in state_shape
        """
        if self.use_flat_obs:
            return self.get_flat_multi_obs()

        # Example: Create a simple occupancy grid
        # You should customize this based on your needs
        
        channels, height, width = self.state_shape
        state = np.zeros(self.state_shape, dtype=np.float32)
        
        # Get vehicle information
        vehicle_ids = traci.vehicle.getIDList()
        
        # Simple approach: encode vehicle density per lane
        # This is a placeholder - implement proper state encoding
        all_lanes = []
        for tls_id in self.controlled_tls_list:
            all_lanes.extend(self.controlled_lanes_per_tls.get(tls_id, []))

        for lane_id in all_lanes:
            try:
                # Get vehicles on this lane
                vehicles_on_lane = traci.lane.getLastStepVehicleNumber(lane_id)
                # Normalize and encode into state
                # ... (implement proper state encoding here)
            except:
                pass
        
        # Alternative: Use waiting time and queue length as features
        # This creates a simplified 1-channel representation
        lane_features = []
        for lane_id in all_lanes[:min(len(all_lanes), height)]:
            try:
                waiting_time = traci.lane.getWaitingTime(lane_id)
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                lane_features.append([waiting_time, queue_length])
            except:
                lane_features.append([0.0, 0.0])
        
        # Normalize and reshape into state tensor
        # This is a placeholder - implement proper preprocessing
        state[0, :len(lane_features), :2] = np.array(lane_features)
        
        return state

    def get_flat_obs(self, tls_id: str) -> np.ndarray:
        """
        Build a flat observation vector for ABSTLight.

        Layout:
        - Queue features (normalized) for first K lanes
        - Waiting-time features (normalized) for first K lanes
        - Current phase one-hot features
        - Zero-padding if OBS_DIM is larger than populated features

        Returns:
            np.ndarray of shape (single_obs_dim,)
        """
        if self.obs_dim is None:
            raise ValueError("Flat observation requested but obs_dim is not configured.")

        obs = np.zeros((self._single_obs_dim,), dtype=np.float32)
        lanes = self.controlled_lanes_per_tls.get(tls_id, [])
        tls_idx = self.controlled_tls_list.index(tls_id)

        # Reserve the tail for one-hot phase encoding.
        phase_dim = min(max(self.configured_num_actions, 1), self._single_obs_dim)
        remaining = self._single_obs_dim - phase_dim
        lane_slots = max(remaining // 2, 0)

        # Fill queue and waiting features from the first lane_slots controlled lanes.
        for idx, lane_id in enumerate(lanes[:lane_slots]):
            try:
                queue_length = float(traci.lane.getLastStepHaltingNumber(lane_id))
                waiting_time = float(traci.lane.getWaitingTime(lane_id))
            except Exception:
                queue_length = 0.0
                waiting_time = 0.0

            # Conservative normalization to keep values numerically stable.
            obs[idx] = np.clip(queue_length / self.obs_queue_normalization, 0.0, 1.0)
            obs[lane_slots + idx] = np.clip(waiting_time / self.obs_waiting_time_normalization, 0.0, 1.0)

        # One-hot phase at the end of vector.
        if phase_dim > 0:
            phase_index = int(self.current_phase[tls_idx]) % phase_dim
            obs[self._single_obs_dim - phase_dim + phase_index] = 1.0

        return obs

    def get_flat_multi_obs(self) -> np.ndarray:
        """
        Build multi-intersection flat observations for ABSTLight.

        Returns:
            np.ndarray of shape (N_agents, obs_dim)
        """
        if len(self._obs_queues) != self.num_agents:
            self._obs_queues = [
                collections.deque(maxlen=self._obs_stack_size)
                for _ in range(self.num_agents)
            ]

        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        for idx, tls_id in enumerate(self.controlled_tls_list):
            raw_obs = self.get_flat_obs(tls_id)
            self._obs_queues[idx].append(raw_obs.copy())

            # If queue is not full (edge-case), pre-fill with the first frame.
            while len(self._obs_queues[idx]) < self._obs_stack_size:
                self._obs_queues[idx].appendleft(raw_obs.copy())

            # Concatenate oldest -> newest to form stacked observation.
            obs[idx] = np.concatenate(list(self._obs_queues[idx]), axis=0)
        return obs

    def _get_intersection_throughput_count(self, tls_id: str) -> float:
        """
        Count destination arrival events for one intersection in the last SUMO step.

        Exploit guard:
        - `getLastStepVehicleNumber(loop_id)` can report a vehicle repeatedly when
          it is stopped on top of a loop.
        - To avoid rewarding parked vehicles, only count loop hits when the loop's
          mean speed indicates movement through the detector.

        Args:
            tls_id: Traffic light ID.

        Returns:
            Destination arrival count for this TLS during the last step.
        """
        passed = 0.0
        for loop_id in self.induction_loops_per_tls.get(tls_id, []):
            try:
                mean_speed = float(traci.inductionloop.getLastStepMeanSpeed(loop_id))
                if mean_speed <= self.speed_threshold:
                    continue

                passed += float(traci.inductionloop.getLastStepVehicleNumber(loop_id))
            except Exception:
                continue

        return passed
    
    def compute_reward(self) -> np.ndarray:
        """
        Compute normalized hybrid reward for each controlled intersection.

        Components:
        - Destination arrival reward from e1 induction loops on local incoming lanes
        - Local capped max-pressure penalty (incoming - outgoing, floor at 0)
        - Clipped insertion-delay penalty from pending departures
        - Phase-change penalty to reduce flickering

        Returns:
            np.ndarray of shape (num_agents,)
        """
        rewards = np.zeros((self.num_agents,), dtype=np.float32)
        total_waiting_time = 0.0

        # Global insertion proxy (vehicles waiting to enter network).
        pending_count = 0.0
        try:
            pending = traci.simulation.getPendingVehicles()
            pending_count = float(len(pending)) if hasattr(pending, '__len__') else float(pending)
        except Exception:
            pending_count = 0.0

        # Normalize by agent count and capacity so scale remains comparable to throughput,
        # then clip the weighted penalty to avoid overwhelming early-stage learning.
        per_agent_pending = pending_count / float(max(self.num_agents, 1))
        insertion_signal = per_agent_pending / self.pressure_normalization_capacity
        insertion_component = self.reward_weights['insertion_penalty'] * insertion_signal
        insertion_component = max(insertion_component, self.insertion_penalty_clip)

        self.last_pending_count = float(pending_count)
        self.last_insertion_signal = float(insertion_signal)
        self.last_insertion_component = float(insertion_component)

        for idx, tls_id in enumerate(self.controlled_tls_list):
            throughput = self._get_intersection_throughput_count(tls_id)

            incoming_vehicles = 0.0
            for lane_id in self.controlled_lanes_per_tls.get(tls_id, []):
                try:
                    incoming_vehicles += float(traci.lane.getLastStepVehicleNumber(lane_id))
                    total_waiting_time += float(traci.lane.getWaitingTime(lane_id))
                except Exception:
                    continue

            outgoing_vehicles = 0.0
            for lane_id in self.outgoing_lanes_per_tls.get(tls_id, []):
                try:
                    outgoing_vehicles += float(traci.lane.getLastStepVehicleNumber(lane_id))
                except Exception:
                    continue

            pressure = max(incoming_vehicles - outgoing_vehicles, 0.0)
            normalized_pressure = pressure / self.pressure_normalization_capacity

            phase_change_component = 0.0
            if idx < len(self.phase_changed_this_step) and self.phase_changed_this_step[idx] > 0.0:
                phase_change_component = self.phase_change_penalty

            rewards[idx] = (
                self.reward_weights['throughput'] * throughput
                + self.reward_weights['pressure'] * normalized_pressure
                + insertion_component
                + phase_change_component
            )

        # Track global statistics
        self.total_waiting_time += total_waiting_time
        self.total_vehicles = len(traci.vehicle.getIDList())

        return rewards
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the environment state.
        
        Returns:
            Dictionary with environment statistics
        """
        info = {
            'step': self.current_step,
            'current_phase': self.current_phase.tolist(),
            'num_actions': self.num_actions,
            'available_phase_count': self.available_phase_count,
            'total_waiting_time': self.total_waiting_time,
            'total_vehicles': self.total_vehicles,
            'time_since_phase_change': self.time_since_last_change.tolist(),
            'safety_fallback_count': int(self._fallback_triggered_count),
            'num_agents': self.num_agents,
            'tls_ids': self.controlled_tls_list,
            'reward_debug': {
                'pending_count': float(self.last_pending_count),
                'insertion_signal': float(self.last_insertion_signal),
                'insertion_component': float(self.last_insertion_component),
            },
            'teleport_count': int(self.episode_teleports),
        }
        
        # Add per-TLS lane statistics
        tls_lane_stats = {}
        for tls_id in self.controlled_tls_list:
            lane_stats = {}
            for lane_id in self.controlled_lanes_per_tls.get(tls_id, []):
                try:
                    lane_stats[lane_id] = {
                        'waiting_time': traci.lane.getWaitingTime(lane_id),
                        'queue_length': traci.lane.getLastStepHaltingNumber(lane_id),
                        'mean_speed': traci.lane.getLastStepMeanSpeed(lane_id)
                    }
                except Exception:
                    continue
            tls_lane_stats[tls_id] = lane_stats

        info['tls_lane_stats'] = tls_lane_stats
        
        return info
    
    def close(self):
        """
        Close the SUMO simulation and clean up.
        """
        if self.is_connected:
            traci.close()
            self.is_connected = False

    def detect_action_space(self) -> int:
        """
        Probe SUMO once and return the effective action count for the controlled TLS.

        Returns:
            Effective number of actions after applying config cap
        """
        if self.is_connected:
            traci.close()
            self.is_connected = False

        self.start_simulation()
        detected_actions = self.num_actions
        self.close()
        return detected_actions

    def detect_num_agents(self) -> int:
        """
        Probe SUMO once and return the number of controllable traffic lights.

        Returns:
            Number of TLS IDs discovered in the active SUMO scenario.
        """
        if self.is_connected:
            traci.close()
            self.is_connected = False

        self.start_simulation()
        detected_agents = self.num_agents
        self.close()
        return detected_agents

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Build an intersection adjacency matrix from the active SUMO topology.

        Adjacency rule:
            TLS i is connected to TLS j if at least one outgoing edge from i
            is an incoming controlled edge of j. The result is symmetrized.

        Returns:
            np.ndarray of shape (N_agents, N_agents), dtype float32
        """
        if self.is_connected:
            was_connected = True
        else:
            was_connected = False
            self.start_simulation()

        try:
            tls_ids = list(self.controlled_tls_list)
            if not tls_ids:
                return np.eye(1, dtype=np.float32)

            incoming_edges: Dict[str, set] = {}
            outgoing_edges: Dict[str, set] = {}

            for tls_id in tls_ids:
                lanes = self.controlled_lanes_per_tls.get(tls_id, [])
                incoming = set()
                for lane_id in lanes:
                    try:
                        incoming.add(traci.lane.getEdgeID(lane_id))
                    except Exception:
                        continue
                incoming_edges[tls_id] = incoming

                outgoing = set()
                links = traci.trafficlight.getControlledLinks(tls_id)
                for signal_links in links:
                    for link in signal_links:
                        if not link or len(link) < 2:
                            continue
                        out_lane = link[1]
                        if not out_lane:
                            continue
                        try:
                            outgoing.add(traci.lane.getEdgeID(out_lane))
                        except Exception:
                            continue
                outgoing_edges[tls_id] = outgoing

            N = len(tls_ids)
            adj = np.zeros((N, N), dtype=np.float32)
            for i, tls_i in enumerate(tls_ids):
                for j, tls_j in enumerate(tls_ids):
                    if i == j:
                        continue
                    if outgoing_edges[tls_i].intersection(incoming_edges[tls_j]):
                        adj[i, j] = 1.0

            # Undirected topology improves stability for shared-neighbour message passing.
            adj = np.maximum(adj, adj.T)

            # Fallback for disconnected extraction: keep non-empty graph signal.
            if not np.any(adj):
                return np.eye(N, dtype=np.float32)

            return adj
        finally:
            if not was_connected:
                self.close()

    def _resolve_net_file_path(self) -> str:
        """
        Resolve the net-file path from the provided .sumocfg.

        Returns:
            Absolute net.xml path if found, otherwise empty string.
        """
        try:
            cfg_path = Path(self.sumo_config)
            tree = ET.parse(cfg_path)
            root = tree.getroot()
            input_elem = root.find('input')
            if input_elem is None:
                return ''
            net_elem = input_elem.find('net-file')
            if net_elem is None:
                return ''
            net_value = net_elem.attrib.get('value', '')
            if not net_value:
                return ''
            return str((cfg_path.parent / net_value).resolve())
        except Exception:
            return ''

    def _get_phase_count(self, tls_id: str) -> int:
        """
        Get the number of phases in the currently active TLS program.

        Args:
            tls_id: Traffic light ID

        Returns:
            Number of phases (at least 1)
        """
        try:
            phase_count = traci.trafficlight.getPhaseNumber(tls_id)
            if phase_count > 0:
                return phase_count
        except Exception:
            pass

        try:
            current_program = traci.trafficlight.getProgram(tls_id)
            logics = traci.trafficlight.getAllProgramLogics(tls_id)

            # Prefer the active program logic if present.
            for logic in logics:
                logic_program = getattr(logic, 'programID', None)
                phases = getattr(logic, 'phases', [])
                if logic_program == current_program and len(phases) > 0:
                    return len(phases)

            for logic in logics:
                phases = getattr(logic, 'phases', [])
                if len(phases) > 0:
                    return len(phases)
        except Exception:
            pass

        # Fallback for older TraCI APIs.
        try:
            definitions = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
            for logic in definitions:
                phases = getattr(logic, 'phases', [])
                if len(phases) > 0:
                    return len(phases)
        except Exception:
            pass

        return 1
    
    def __del__(self):
        """
        Destructor to ensure SUMO is closed.
        """
        self.close()
