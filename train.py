"""
ABSTLight Training Script

Main training loop for the ABSTLight traffic control agent.
This script initializes the environment, agent, and replay buffer,
then runs the training loop for the specified number of episodes.

Usage:
    python train.py

Make sure to:
1. Set SUMO_HOME environment variable
2. Configure parameters in config.py
3. Ensure SUMO configuration files exist in sumo_files/
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path
import time
from datetime import datetime
import json
import argparse
from types import SimpleNamespace

# Import project modules
from config import ABSTConfig
from environment.traffic_env import TrafficEnvironment
from agent.abst_light_agent import ABSTLightAgent
from memory.replay_buffer import ReplayBuffer


class RuntimeConfig(SimpleNamespace):
    """Mutable per-run config snapshot to avoid mutating class-level config."""

    def to_dict(self):
        return dict(self.__dict__)

    def print_config(self):
        print("=" * 60)
        print("ABSTLight Runtime Configuration")
        print("=" * 60)
        for key, value in self.to_dict().items():
            print(f"{key:.<40} {value}")
        print("=" * 60)


def _config_to_runtime(config_source):
    """Build a runtime copy from a config class/object."""
    values = {}

    # Prefer explicit serializer if present.
    if hasattr(config_source, 'to_dict') and callable(config_source.to_dict):
        values.update(dict(config_source.to_dict()))

    # Include inherited class attributes (e.g., ABSTConfig <- Config).
    config_cls = config_source if isinstance(config_source, type) else type(config_source)
    for cls in reversed(config_cls.mro()):
        if cls is object:
            continue
        for key, value in vars(cls).items():
            if key.startswith('_'):
                continue
            if callable(value):
                continue
            if isinstance(value, (classmethod, staticmethod)):
                continue
            values[key] = value

    # Finally, allow instance-level fields to override class defaults.
    if not isinstance(config_source, type):
        for key, value in vars(config_source).items():
            if key.startswith('_'):
                continue
            if callable(value):
                continue
            values[key] = value

    return RuntimeConfig(**values)


def _extract_episode_number(dirname: str) -> int:
    """Extract episode number from a checkpoint directory name."""
    prefix = "episode_"
    if not dirname.startswith(prefix):
        return -1

    suffix = dirname[len(prefix):]
    return int(suffix) if suffix.isdigit() else -1


def find_latest_checkpoint(models_root: str):
    """
    Return the latest valid checkpoint directory and episode number.

    Args:
        models_root: Root directory containing episode_* checkpoint folders.

    Returns:
        Tuple of (checkpoint_dir, episode_number), or (None, None) if not found.
    """
    root = Path(models_root)
    if not root.exists() or not root.is_dir():
        return None, None

    candidates = []
    for path in root.iterdir():
        if not path.is_dir():
            continue

        episode_num = _extract_episode_number(path.name)
        if episode_num < 0:
            continue

        has_state = (path / f"agent_state_episode_{episode_num}.pth").exists()
        has_network = (
            (path / f"embedding_episode_{episode_num}.pth").exists()
            and (path / f"gcn_mha_episode_{episode_num}.pth").exists()
            and any(path.glob(f"q_head_agent*_net*_episode_{episode_num}.pth"))
        )

        if has_state and has_network:
            candidates.append((episode_num, str(path)))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: item[0], reverse=True)
    latest_episode, latest_dir = candidates[0]
    return latest_dir, latest_episode


class Trainer:
    """
    Training manager for ABSTLight agent.
    
    Handles:
    - Environment initialization
    - Agent initialization
    - Training loop execution
    - Model checkpointing
    - Logging and statistics
    """
    
    def __init__(self, config=ABSTConfig, use_gui=None, num_episodes=None, route_file=None):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration class with hyperparameters
            use_gui: Override USE_GUI setting (True/False/None)
            num_episodes: Override NUM_EPISODES setting (int/None)
            route_file: Optional route file override passed to SUMO CLI (-r)
        """
        self.route_file = route_file
        self._last_logged_route_file = None
        # Use a runtime config snapshot to prevent accidental mutation of
        # class-level ABSTConfig values during a training run.
        self.config = _config_to_runtime(config)
        
        # Override config settings if specified
        if use_gui is not None:
            self.config.USE_GUI = use_gui
        if num_episodes is not None:
            self.config.NUM_EPISODES = num_episodes
        
        # Setup device
        if self.config.USE_CUDA and torch.cuda.is_available():
            self.device = 'cuda'
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print("Using CPU")
        
        # Create directories
        self._setup_directories()
        
        # Initialize components
        print("\n" + "="*60)
        print("Initializing ABSTLight Training")
        print("="*60)
        
        self._init_environment()
        self._init_agent()
        self._init_memory()
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_insertion_component_stats = []
        self.episode_scenario_stats = []
        self.training_start_time = None
        self._prev_rejected_subsets = 0
        self._last_episode_insertion_component_stats = {}
        self._last_episode_scenario_stats = {}

    def _resolve_route_for_episode(self, episode: int):
        """
        Resolve route file and mode for a training episode.

        Schedule (default):
        - Episode 1-500:    sumo_files/osm.moderate.rou.xml
        - Episode 501-850:  sumo_files/osm.peak.rou.xml
        - Episode 851+:     random choice of the two above

        If a route override is provided in Trainer(..., route_file=...),
        that route is used for all episodes.
        """
        if self.route_file:
            return self.route_file, 'override'

        moderate_route = "sumo_files/osm.moderate.rou.xml"
        peak_route = "sumo_files/osm.peak.rou.xml"

        if episode <= 500:
            return moderate_route, 'moderate'
        if episode <= 850:
            return peak_route, 'peak'

        return str(np.random.choice([moderate_route, peak_route])), 'random'

    def _apply_episode_route(self, episode: int):
        """
        Apply scheduled route file to environment before episode reset.
        """
        route_file, route_mode = self._resolve_route_for_episode(episode)
        route_path = Path(route_file)
        if not route_path.exists():
            raise FileNotFoundError(f"Scheduled route file not found: {route_file}")

        self.env.route_file = route_file

        # Always log random mode; otherwise log only when route changes.
        should_log = route_mode == 'random' or route_file != self._last_logged_route_file
        if should_log:
            print(
                f"[Route] Episode {episode}/{self.config.NUM_EPISODES} | "
                f"Mode: {route_mode} | File: {route_file}"
            )
            self._last_logged_route_file = route_file

    def _to_flat_multi_agent_obs(self, state: np.ndarray) -> np.ndarray:
        """
        Convert environment state into ABSTLight input shape (N_agents, obs_dim).

        TrafficEnvironment in ABSTLight returns flat multi-agent observations
        in shape (N_agents, obs_dim), where obs_dim may already include
        temporal stacking (T * raw_obs_dim). For single-agent compatibility,
        a flat (obs_dim,) vector is reshaped to (1, obs_dim).

        Args:
            state: Environment state array.

        Returns:
            np.ndarray of shape (self.num_agents, self.obs_dim).
        """
        arr = np.asarray(state, dtype=np.float32)

        # Native ABSTLight path: environment returns (N_agents, obs_dim).
        if arr.ndim == 2 and arr.shape == (self.num_agents, self.obs_dim):
            return arr

        # Backward-compatible single-intersection path.
        if arr.ndim == 1 and self.num_agents == 1 and arr.shape[0] == self.obs_dim:
            return arr.reshape(1, self.obs_dim)

        raise ValueError(
            f"Observation shape mismatch: got {arr.shape}, expected "
            f"({self.num_agents}, {self.obs_dim})."
        )
        
    def _setup_directories(self):
        """Create necessary directories for saving models and logs."""
        Path(self.config.MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.config.LOG_DIR).mkdir(parents=True, exist_ok=True)
        print(f"Created directories: {self.config.MODEL_SAVE_DIR}, {self.config.LOG_DIR}")
    
    def _init_environment(self):
        """Initialize traffic simulation environment."""
        print("\nInitializing Traffic Environment...")
        
        reward_weights = {
            'throughput': self.config.REWARD_THROUGHPUT_WEIGHT,
            'pressure': self.config.REWARD_PRESSURE_WEIGHT,
            'insertion_penalty': self.config.REWARD_INSERTION_PENALTY_WEIGHT,
            'phase_change': self.config.REWARD_PHASE_CHANGE,
        }
        
        self.env = TrafficEnvironment(
            sumo_config=self.config.SUMO_CONFIG_PATH,
            use_gui=self.config.USE_GUI,
            route_file=self.route_file,
            num_actions=self.config.NUM_ACTIONS,
            state_shape=(self.config.STATE_CHANNELS, self.config.STATE_HEIGHT, self.config.STATE_WIDTH),
            obs_dim=getattr(self.config, 'OBS_DIM', None),
            obs_stack_size=int(getattr(self.config, 'OBS_STACK_SIZE', 1)),
            yellow_duration=self.config.YELLOW_PHASE_DURATION,
            min_green_duration=self.config.MIN_GREEN_DURATION,
            reward_weights=reward_weights,
            max_steps=self.config.MAX_STEPS_PER_EPISODE,
            pressure_normalization_capacity=float(
                getattr(self.config, 'PRESSURE_NORMALIZATION_CAPACITY', 50.0)
            ),
            insertion_penalty_clip=float(
                getattr(self.config, 'INSERTION_PENALTY_CLIP', -5.0)
            ),
            obs_queue_normalization=float(
                getattr(self.config, 'OBS_QUEUE_NORMALIZATION', 50.0)
            ),
            obs_waiting_time_normalization=float(
                getattr(self.config, 'OBS_WAITING_TIME_NORMALIZATION', 300.0)
            )
        )

        expected_insertion_weight = float(self.config.REWARD_INSERTION_PENALTY_WEIGHT)
        actual_insertion_weight = float(self.env.reward_weights.get('insertion_penalty', 0.0))
        if abs(actual_insertion_weight - expected_insertion_weight) > 1e-12:
            raise ValueError(
                "Reward weight injection mismatch for insertion_penalty: "
                f"expected {expected_insertion_weight}, got {actual_insertion_weight}."
            )
        print(
            "  Reward Weights Injected: "
            f"throughput={self.env.reward_weights['throughput']}, "
            f"pressure={self.env.reward_weights['pressure']}, "
            f"insertion_penalty={actual_insertion_weight}, "
            f"phase_change={self.env.reward_weights['phase_change']}"
        )

        # Detect the effective action space from the active SUMO TLS program.
        self.num_actions = self.env.detect_action_space()
        
        print(f"  SUMO Config: {self.config.SUMO_CONFIG_PATH}")
        if self.route_file:
            print(f"  Route Override: {self.route_file}")
        print(f"  State Shape: ({self.config.STATE_CHANNELS}, {self.config.STATE_HEIGHT}, {self.config.STATE_WIDTH})")
        if hasattr(self.config, 'OBS_DIM'):
            self.obs_dim = int(self.config.OBS_DIM)
        else:
            self.obs_dim = int(self.config.STATE_CHANNELS * self.config.STATE_HEIGHT * self.config.STATE_WIDTH)
        detected_agents = int(self.env.detect_num_agents())
        configured_agents = int(getattr(self.config, 'NUM_AGENTS', detected_agents))
        if configured_agents != detected_agents:
            print(
                f"[INFO] NUM_AGENTS auto-synced from {configured_agents} to "
                f"SUMO TLS count {detected_agents}."
            )

        # Keep runtime config aligned with the active SUMO scenario so all
        # downstream logs/checkpoints reflect the real multi-agent count.
        self.config.NUM_AGENTS = detected_agents
        self.num_agents = detected_agents
        print(f"  ABST Obs Dim (from config): {self.obs_dim}")
        print(f"  ABST Num Agents: {self.num_agents}")
        print(f"  Configured Actions: {self.config.NUM_ACTIONS}")
        print(f"  Effective Actions: {self.num_actions}")
    
    def _init_agent(self):
        """Initialize ABSTLight agent."""
        print("\nInitializing ABSTLight Agent...")

        has_lr = hasattr(self.config, 'LEARNING_RATE')
        has_gamma = hasattr(self.config, 'GAMMA')
        has_threshold = hasattr(self.config, 'ERROR_THRESHOLD')
        has_reward_scale = hasattr(self.config, 'REWARD_SCALE_DIVISOR')
        has_dynamic_threshold = hasattr(self.config, 'USE_DYNAMIC_ERROR_THRESHOLD')
        has_relative_ratio = hasattr(self.config, 'RELATIVE_ERROR_THRESHOLD_RATIO')
        has_min_target_scale = hasattr(self.config, 'MIN_TARGET_SCALE_FOR_THRESHOLD')
        has_resampling = hasattr(self.config, 'MAX_RESAMPLING_ATTEMPTS')

        learning_rate = float(getattr(self.config, 'LEARNING_RATE', 1e-4))
        gamma = float(getattr(self.config, 'GAMMA', 0.95))
        error_threshold = float(getattr(self.config, 'ERROR_THRESHOLD', 150.0))
        reward_scale_divisor = float(getattr(self.config, 'REWARD_SCALE_DIVISOR', 1000.0))
        use_dynamic_error_threshold = bool(getattr(self.config, 'USE_DYNAMIC_ERROR_THRESHOLD', True))
        relative_error_threshold_ratio = float(getattr(self.config, 'RELATIVE_ERROR_THRESHOLD_RATIO', 0.05))
        min_target_scale_for_threshold = float(getattr(self.config, 'MIN_TARGET_SCALE_FOR_THRESHOLD', 1.0))
        max_resampling_attempts = int(getattr(self.config, 'MAX_RESAMPLING_ATTEMPTS', 5))

        self.resolved_rl_hparams = {
            'learning_rate': learning_rate,
            'gamma': gamma,
            'error_threshold': error_threshold,
            'reward_scale_divisor': reward_scale_divisor,
            'use_dynamic_error_threshold': use_dynamic_error_threshold,
            'relative_error_threshold_ratio': relative_error_threshold_ratio,
            'min_target_scale_for_threshold': min_target_scale_for_threshold,
            'max_resampling_attempts': max_resampling_attempts,
            'source': {
                'learning_rate': 'config.LEARNING_RATE' if has_lr else 'fallback(1e-4)',
                'gamma': 'config.GAMMA' if has_gamma else 'fallback(0.95)',
                'error_threshold': 'config.ERROR_THRESHOLD' if has_threshold else 'fallback(150.0)',
                'reward_scale_divisor': (
                    'config.REWARD_SCALE_DIVISOR' if has_reward_scale else 'fallback(1000.0)'
                ),
                'use_dynamic_error_threshold': (
                    'config.USE_DYNAMIC_ERROR_THRESHOLD' if has_dynamic_threshold else 'fallback(True)'
                ),
                'relative_error_threshold_ratio': (
                    'config.RELATIVE_ERROR_THRESHOLD_RATIO' if has_relative_ratio else 'fallback(0.05)'
                ),
                'min_target_scale_for_threshold': (
                    'config.MIN_TARGET_SCALE_FOR_THRESHOLD' if has_min_target_scale else 'fallback(1.0)'
                ),
                'max_resampling_attempts': (
                    'config.MAX_RESAMPLING_ATTEMPTS' if has_resampling else 'fallback(5)'
                ),
            },
        }

        print("  Resolved RL Hyperparameters:")
        print(
            f"    learning_rate={learning_rate} "
            f"[{self.resolved_rl_hparams['source']['learning_rate']}]"
        )
        print(
            f"    gamma={gamma} "
            f"[{self.resolved_rl_hparams['source']['gamma']}]"
        )
        print(
            f"    error_threshold={error_threshold} "
            f"[{self.resolved_rl_hparams['source']['error_threshold']}]"
        )
        print(
            f"    reward_scale_divisor={reward_scale_divisor} "
            f"[{self.resolved_rl_hparams['source']['reward_scale_divisor']}]"
        )
        print(
            f"    use_dynamic_error_threshold={use_dynamic_error_threshold} "
            f"[{self.resolved_rl_hparams['source']['use_dynamic_error_threshold']}]"
        )
        print(
            f"    relative_error_threshold_ratio={relative_error_threshold_ratio} "
            f"[{self.resolved_rl_hparams['source']['relative_error_threshold_ratio']}]"
        )
        print(
            f"    min_target_scale_for_threshold={min_target_scale_for_threshold} "
            f"[{self.resolved_rl_hparams['source']['min_target_scale_for_threshold']}]"
        )
        print(
            f"    max_resampling_attempts={max_resampling_attempts} "
            f"[{self.resolved_rl_hparams['source']['max_resampling_attempts']}]"
        )

        # Use SUMO topology-derived adjacency to preserve ABSTLight communication.
        adj_matrix = self.env.get_adjacency_matrix().astype(np.float32)
        if adj_matrix.shape != (self.num_agents, self.num_agents):
            raise ValueError(
                f"Adjacency shape mismatch: got {adj_matrix.shape}, expected "
                f"({self.num_agents}, {self.num_agents})."
            )

        self.agent = ABSTLightAgent(
            obs_dim=self.obs_dim,
            num_agents=self.num_agents,
            num_actions=self.num_actions,
            adj_matrix=adj_matrix,
            embed_dim=self.config.EMBED_DIM,
            num_heads=self.config.NUM_HEADS,
            num_gcn_layers=self.config.NUM_GCN_LAYERS,
            gcn_dropout=self.config.GCN_DROPOUT,
            hidden_dim=self.config.Q_HEAD_HIDDEN_DIM,
            N=self.config.N_NETWORKS,
            M=self.config.M_SUBSET_SIZE,
            learning_rate=learning_rate,
            gamma=gamma,
            error_threshold=error_threshold,
            reward_scale_divisor=reward_scale_divisor,
            use_dynamic_error_threshold=use_dynamic_error_threshold,
            relative_error_threshold_ratio=relative_error_threshold_ratio,
            min_target_scale_for_threshold=min_target_scale_for_threshold,
            max_resampling_attempts=max_resampling_attempts,
            epsilon_start=self.config.EPSILON_START,
            epsilon_end=self.config.EPSILON_END,
            epsilon_decay=self.config.EPSILON_DECAY,
            device=self.device
        )

        print(f"  Ensemble Size: N={self.config.N_NETWORKS}")
        print(f"  Subset Size: M={self.config.M_SUBSET_SIZE}")
        print(f"  Error Threshold: {self.config.ERROR_THRESHOLD}")
    
    def _init_memory(self):
        """Initialize replay buffer."""
        print("\nInitializing Replay Buffer...")

        # ABSTLight replay state shape: (N_agents, obs_dim)
        state_shape = (self.num_agents, self.obs_dim)
        
        self.memory = ReplayBuffer(
            capacity=self.config.MEMORY_SIZE,
            state_shape=state_shape,
            device=self.device
        )
        
        print(f"  Capacity: {self.config.MEMORY_SIZE}")
        print(f"  Min Size for Training: {self.config.MIN_MEMORY_SIZE}")

    def _resume_from_latest_checkpoint(self) -> int:
        """
        Resume training from the latest available checkpoint.

        Returns:
            The next episode number to train.
        """
        checkpoint_dir, checkpoint_episode = find_latest_checkpoint(self.config.MODEL_SAVE_DIR)
        if checkpoint_dir is None or checkpoint_episode is None:
            print("No checkpoint found. Starting training from episode 1.")
            return 1

        print(f"Found latest checkpoint: {checkpoint_dir}")
        self.agent.load(checkpoint_dir, episode=checkpoint_episode)

        resumed_episode = int(getattr(self.agent, 'episode_count', checkpoint_episode))
        if resumed_episode <= 0:
            resumed_episode = checkpoint_episode

        next_episode = resumed_episode + 1
        print(f"Resuming training from episode {next_episode}.")
        return next_episode
    
    def train(self):
        """
        Main training loop.
        
        Executes training for the configured number of episodes.
        """
        self.training_start_time = time.time()
        
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        self.config.print_config()

        start_episode = self._resume_from_latest_checkpoint()
        # Establish baseline so per-episode rejected subsets are computed as deltas.
        self._prev_rejected_subsets = int(
            self.agent.get_statistics()['updater_stats'].get('rejected_subsets', 0)
        )
        if start_episode > self.config.NUM_EPISODES:
            print(
                f"Checkpoint already reached episode {start_episode - 1}, "
                f"which meets/exceeds NUM_EPISODES={self.config.NUM_EPISODES}."
            )
            self.env.close()
            return
        
        for episode in range(start_episode, self.config.NUM_EPISODES + 1):
            episode_reward, episode_length = self._train_episode(episode)
            
            # Store statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_insertion_component_stats.append(
                dict(self._last_episode_insertion_component_stats)
            )
            self.episode_scenario_stats.append(
                dict(self._last_episode_scenario_stats)
            )
            
            # Print episode summary
            self._print_episode_summary(episode, episode_reward, episode_length)
            
            # Decay epsilon for next episode
            self.agent.decay_epsilon()
            
            # Save model checkpoint
            if episode % self.config.SAVE_FREQUENCY == 0:
                self._save_checkpoint(episode)
            
            # Save training log
            if episode % 10 == 0:
                self._save_training_log()
        
        # Final save
        self._save_checkpoint(self.config.NUM_EPISODES)
        self._save_training_log()
        
        print("\n" + "="*60)
        print("Training Completed!")
        print("="*60)
        
        # Close environment
        self.env.close()
    
    def _train_episode(self, episode: int):
        """
        Train for one episode.
        
        Args:
            episode: Current episode number
            
        Returns:
            Tuple of (total_reward, episode_length)
        """
        # Apply per-episode route schedule, then reset environment.
        self._apply_episode_route(episode)

        # Reset environment
        state = self.env.reset()
        state = self._to_flat_multi_agent_obs(state)
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        insertion_component_trace = []
        scenario_trace = []
        context_tag_trace = []
        dynamic_accident_enabled = False
        dynamic_accident_triggered = False
        dynamic_accident_activation_step = None
        sampled_scenario = None
        blocked_edge = ""
        blocked_lane = ""
        progress_interval = 900
        next_progress_step = progress_interval
        
        while not done:
            # Select one action per intersection via ABSTLight ensemble.
            state_tensor = torch.from_numpy(state).float()
            actions = self.agent.select_all_actions(state_tensor, mode='voting')
            actions_np = np.asarray(actions, dtype=np.int64)
            
            # Execute action
            next_state, reward, done, info = self.env.step(actions_np)
            next_state = self._to_flat_multi_agent_obs(next_state)
            reward = np.asarray(reward, dtype=np.float32).reshape(self.num_agents)

            reward_debug = info.get('reward_debug', {})
            if 'insertion_component' in reward_debug:
                insertion_component_trace.append(float(reward_debug['insertion_component']))

            scenario_name = info.get('scenario')
            if scenario_name is not None:
                scenario_name = str(scenario_name)
                scenario_trace.append(scenario_name)
                if sampled_scenario is None:
                    sampled_scenario = scenario_name

            context_tag = info.get('context_tag')
            if isinstance(context_tag, (list, tuple)) and len(context_tag) == 2:
                context_tag_trace.append([float(context_tag[0]), float(context_tag[1])])

            if 'dynamic_accident_enabled' in info:
                dynamic_accident_enabled = bool(info.get('dynamic_accident_enabled'))
            if bool(info.get('dynamic_accident_triggered', False)):
                dynamic_accident_triggered = True

            activation_step = info.get('dynamic_accident_activation_step')
            if activation_step is not None:
                dynamic_accident_activation_step = int(activation_step)

            edge = info.get('blocked_edge')
            lane = info.get('blocked_lane')
            if edge:
                blocked_edge = str(edge)
            if lane:
                blocked_lane = str(lane)

            # Periodic progress output based on SUMO simulation time.
            current_sumo_step = info.get('step', 0)
            while current_sumo_step >= next_progress_step and next_progress_step <= self.config.MAX_STEPS_PER_EPISODE:
                print(
                    f"[Progress] Episode {episode}/{self.config.NUM_EPISODES} | "
                    f"SUMO Time: {next_progress_step}s/{self.config.MAX_STEPS_PER_EPISODE}s"
                )
                next_progress_step += progress_interval
            
            # Store transition in memory
            self.memory.push(
                state,
                actions_np,
                reward,
                next_state,
                done
            )
            
            # Update agent if memory is ready
            if self.memory.is_ready(self.config.MIN_MEMORY_SIZE):
                if episode_length % self.config.UPDATE_FREQUENCY == 0:
                    # Sample batch from memory
                    batch = self.memory.sample(self.config.BATCH_SIZE)

                    states, actions_b, rewards_b, next_states, dones = batch

                    # Update ABSTLight model.
                    update_results = self.agent.update(
                        states.float(),
                        actions_b,
                        rewards_b,
                        next_states.float(),
                        dones.float(),
                    )
            
            # Update state
            state = next_state
            episode_reward += float(np.sum(reward))
            episode_length += 1
        
        # Increment episode counter
        if hasattr(self.agent, 'increment_episode'):
            self.agent.increment_episode()
        else:
            self.agent.episode_count += 1

        if insertion_component_trace:
            trace = np.asarray(insertion_component_trace, dtype=np.float32)
            self._last_episode_insertion_component_stats = {
                'episode': int(episode),
                'steps': int(trace.shape[0]),
                'mean': float(np.mean(trace)),
                'min': float(np.min(trace)),
                'max': float(np.max(trace)),
                'last': float(trace[-1]),
                'trace': [float(v) for v in insertion_component_trace],
            }
        else:
            self._last_episode_insertion_component_stats = {
                'episode': int(episode),
                'steps': 0,
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'last': 0.0,
                'trace': [],
            }

        if scenario_trace:
            scenario_counts = {
                'sunny': int(sum(1 for s in scenario_trace if s == 'sunny')),
                'rainy': int(sum(1 for s in scenario_trace if s == 'rainy')),
                'accident': int(sum(1 for s in scenario_trace if s == 'accident')),
            }
            final_scenario = scenario_trace[-1]
        else:
            scenario_counts = {'sunny': 0, 'rainy': 0, 'accident': 0}
            final_scenario = None

        first_context_tag = context_tag_trace[0] if context_tag_trace else None
        last_context_tag = context_tag_trace[-1] if context_tag_trace else None

        self._last_episode_scenario_stats = {
            'episode': int(episode),
            'steps': int(episode_length),
            'sampled_scenario': sampled_scenario,
            'final_scenario': final_scenario,
            'scenario_counts': scenario_counts,
            'first_context_tag': first_context_tag,
            'last_context_tag': last_context_tag,
            'dynamic_accident_enabled': bool(dynamic_accident_enabled),
            'dynamic_accident_triggered': bool(dynamic_accident_triggered),
            'dynamic_accident_activation_step': dynamic_accident_activation_step,
            'blocked_edge': blocked_edge,
            'blocked_lane': blocked_lane,
        }
        
        return episode_reward, episode_length
    
    def test(self, num_episodes=10, model_path=None):
        """
        Test the trained agent.
        
        Args:
            num_episodes: Number of test episodes to run
            model_path: Path to saved model checkpoint (optional)
        """
        print("\n" + "="*60)
        print("Starting Testing")
        print("="*60)
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.agent.load(model_path)
        else:
            print("Testing with current agent state")
        
        # Set agent to evaluation mode
        self.agent.eval_mode()
        self.agent.set_epsilon(0.0)  # No exploration during testing
        
        test_rewards = []
        test_lengths = []
        
        for episode in range(1, num_episodes + 1):
            episode_reward, episode_length = self._test_episode(episode)
            test_rewards.append(episode_reward)
            test_lengths.append(episode_length)
            
            # Print test episode summary
            avg_reward = np.mean(test_rewards)
            print(f"\nTest Episode {episode}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f}  |  Avg: {avg_reward:.2f}")
            print(f"  Length: {episode_length}")
        
        print("\n" + "="*60)
        print("Testing Completed!")
        print("="*60)
        print(f"Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
        print(f"Average Length: {np.mean(test_lengths):.2f} ± {np.std(test_lengths):.2f}")
        
        # Close environment
        self.env.close()
        
        return test_rewards, test_lengths
    
    def _test_episode(self, episode: int):
        """
        Run one test episode.
        
        Args:
            episode: Current episode number
            
        Returns:
            Tuple of (total_reward, episode_length)
        """
        # Reset environment
        state = self.env.reset()
        state = self._to_flat_multi_agent_obs(state)
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Select action (no exploration)
            state_tensor = torch.from_numpy(state).float()
            actions = self.agent.select_all_actions(state_tensor, mode='voting')
            actions_np = np.asarray(actions, dtype=np.int64)
            
            # Execute action
            next_state, reward, done, info = self.env.step(actions_np)
            next_state = self._to_flat_multi_agent_obs(next_state)
            reward = np.asarray(reward, dtype=np.float32).reshape(self.num_agents)
            
            # Update state
            state = next_state
            episode_reward += float(np.sum(reward))
            episode_length += 1
        
        return episode_reward, episode_length
    
    def _print_episode_summary(self, episode: int, reward: float, length: int):
        """
        Print summary of episode performance.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length (steps)
        """
        # Calculate statistics
        avg_reward = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
        
        # Get agent statistics
        agent_stats = self.agent.get_statistics()
        epsilon = agent_stats['epsilon']
        updater_stats = agent_stats['updater_stats']
        
        # Get memory statistics
        memory_stats = self.memory.get_statistics()
        
        if self.config.VERBOSE:
            print(f"\nEpisode {episode}/{self.config.NUM_EPISODES}")
            print(f"  Reward: {reward:.2f}  |  Avg (100): {avg_reward:.2f}")
            print(f"  Length: {length}  |  Epsilon: {epsilon:.3f}")
            print(f"  Memory: {memory_stats['size']}/{memory_stats['capacity']} ({memory_stats['utilization']*100:.1f}%)")
            
            if updater_stats['total_updates'] > 0:
                cumulative_rejected = int(updater_stats['rejected_subsets'])
                episode_rejected = max(
                    0, cumulative_rejected - self._prev_rejected_subsets
                )
                print(f"  Avg Resampling: {updater_stats['avg_resampling_per_update']:.2f}")
                print(f"  Rejected Subsets: {episode_rejected}")
                print(f"  Cumulative R.S.: {cumulative_rejected}")
                self._prev_rejected_subsets = cumulative_rejected

            insertion_stats = getattr(self, '_last_episode_insertion_component_stats', None)
            if insertion_stats and insertion_stats.get('steps', 0) > 0:
                print(
                    "  Insertion Component: "
                    f"mean={insertion_stats['mean']:.3f}, "
                    f"min={insertion_stats['min']:.3f}, "
                    f"max={insertion_stats['max']:.3f}, "
                    f"last={insertion_stats['last']:.3f}"
                )
        else:
            print(f"Episode {episode}: Reward={reward:.2f}, Avg={avg_reward:.2f}, ε={epsilon:.3f}")
    
    def _save_checkpoint(self, episode: int):
        """
        Save model checkpoint.
        
        Args:
            episode: Current episode number
        """
        checkpoint_dir = os.path.join(self.config.MODEL_SAVE_DIR, f"episode_{episode}")
        self.agent.save(checkpoint_dir, episode)
        print(f"\n[CHECKPOINT] Model saved at episode {episode}")
    
    def _save_training_log(self):
        """Save training statistics to JSON file."""
        log_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_insertion_component_stats': self.episode_insertion_component_stats,
            'episode_scenario_stats': self.episode_scenario_stats,
            'agent_stats': self.agent.get_statistics(),
            'memory_stats': self.memory.get_statistics(),
            'config': self.config.to_dict(),
            'resolved_rl_hyperparameters': getattr(self, 'resolved_rl_hparams', {}),
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.config.LOG_DIR, f"training_log_{timestamp}.json")
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)


def main():
    """
    Main entry point for training.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
                description='ABSTLight Traffic Control Agent - Training and Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train with GUI:        python train.py --gui
  Train 100 episodes:    python train.py --episodes 100
    Resume training:       python train.py
  Test with GUI:         python train.py --test --gui
    Test saved model:      python train.py --test --model model/episode_100
        """
    )
    
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Use SUMO GUI for visualization'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode instead of training'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of episodes to run (overrides config)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to saved model checkpoint for testing'
    )
    
    parser.add_argument(
        '--test-episodes',
        type=int,
        default=10,
        help='Number of episodes for testing (default: 10)'
    )

    parser.add_argument(
        '--route-file',
        type=str,
        default=None,
        help='Optional SUMO route file override (passed to SUMO -r)'
    )
    
    args = parser.parse_args()
    
    # Check SUMO installation
    if 'SUMO_HOME' not in os.environ:
        print("ERROR: SUMO_HOME environment variable not set!")
        print("Please install SUMO and set SUMO_HOME.")
        print("  Windows: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo")
        print("  Linux/Mac: export SUMO_HOME=/usr/share/sumo")
        sys.exit(1)
    
    # Check if SUMO config exists
    if not os.path.exists(ABSTConfig.SUMO_CONFIG_PATH):
        print(f"WARNING: SUMO config not found at {ABSTConfig.SUMO_CONFIG_PATH}")
        print("Please ensure your SUMO configuration file exists.")
        print("You may need to update ABSTConfig.SUMO_CONFIG_PATH in config.py")
    
    # Initialize trainer with optional overrides
    trainer = Trainer(
        ABSTConfig,
        use_gui=args.gui,
        num_episodes=args.episodes,
        route_file=args.route_file
    )
    
    # Run training or testing
    try:
        if args.test:
            # Test mode
            print(f"\n{'='*60}")
            print(f"MODE: Testing {'with GUI' if args.gui else ''}")
            print(f"{'='*60}\n")
            trainer.test(
                num_episodes=args.test_episodes,
                model_path=args.model
            )
        else:
            # Training mode
            print(f"\n{'='*60}")
            print(f"MODE: Training {'with GUI' if args.gui else ''}")
            print(f"{'='*60}\n")
            trainer.train()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        trainer.env.close()
    except Exception as e:
        print(f"\n\nError during {'testing' if args.test else 'training'}: {e}")
        import traceback
        traceback.print_exc()
        trainer.env.close()


if __name__ == "__main__":
    main()
