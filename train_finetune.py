"""
train_finetune.py  —  3-in-1 Universal Model Fine-Tuning Pipeline
==================================================================

Fine-tunes the pre-trained ABSTLight base model (1000-episode Sunny expert)
for an additional 500 episodes using domain randomisation over three scenarios:

    Sunny / Normal (50%)  →  context tag [0, 0]
    Rainy          (30%)  →  context tag [1, 0]
    Accident       (20%)  →  context tag [0, 1]

Design goals
------------
* **Zero overwriting** — the original ``train.py``, ``traffic_env.py``, and
  ``model/`` checkpoints are never touched.
* **Gateway check** — the script refuses to start unless the surgically
  converted ``finetune_ready_embedding.pth`` exists.  Run ``weight_converter.py``
  first to produce it.
* **Isolated checkpoint directory** — all fine-tuning checkpoints are written
  to ``model_finetune/``, completely separate from ``model/``.
* **Fresh replay buffer** — no 80-D experiences from base training are carried
  over; a brand-new 82-D buffer (capacity 20 000) is created.
* **Catastrophic-forgetting protection**:
    - Lower learning rate (5e-5 vs 1e-4) protects the GCN + MHA backbone.
    - Epsilon starts at 0.20 (moderate exploration) and decays to ≈ 0.01
      over 500 episodes.

Prerequisites
-------------
1.  ``model/episode_1000/`` — full base-model checkpoint directory.
2.  ``model/episode_1000/finetune_ready_embedding.pth`` — produced by
    running ``python weight_converter.py``.

Usage
-----
    # First-time run (loads base model weights)
    python train_finetune.py

    # Resume an interrupted fine-tuning run
    python train_finetune.py

    # With GUI
    python train_finetune.py --gui

    # Override fine-tuning episode count
    python train_finetune.py --episodes 300
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config import ABSTConfig
from agent.abst_light_agent import ABSTLightAgent
from environment.traffic_env_mixed import MixedTrafficEnv
from memory.replay_buffer import ReplayBuffer
from train import Trainer, find_latest_checkpoint


# ============================================================================
# Fine-Tuning Configuration
# ============================================================================

class FineTuneConfig(ABSTConfig):
    """
    ABSTConfig variant for the 500-episode 3-in-1 fine-tuning phase.

    Key changes vs. ABSTConfig
    --------------------------
    OBS_DIM          80D stacked + 2D context = 82D total (agent input)
    LEARNING_RATE    5e-5   ← smaller to protect the pre-trained backbone
    EPSILON_START    0.20   ← moderate exploration (not starting from scratch)
    EPSILON_END      0.01
    EPSILON_DECAY    calculated so epsilon reaches EPSILON_END at episode 500
    NUM_EPISODES     500
    MODEL_SAVE_DIR   model_finetune/   ← isolated from base model/
    LOG_DIR          logs/finetune/
    MEMORY_SIZE      20000             ← fresh 82-D buffer (no 80-D reuse)
    SAVE_FREQUENCY   50                ← checkpoint every 50 episodes
    """

    # ---- Observation ----
    # The agent sees 82-D observations (80D spatial-temporal + 2D context tag).
    # The environment's obs_dim is still 80 internally; total_obs_dim = 82.
    OBS_DIM = 82

    # ---- Fine-tuning hyperparameters ----
    LEARNING_RATE = 5e-5

    # Epsilon: 0.20 → 0.01 over exactly 500 episodes
    # Derivation: EPSILON_START * EPSILON_DECAY^500 = EPSILON_END
    #   0.20 * d^500 = 0.01  →  d = (0.01/0.20)^(1/500) ≈ 0.9940
    EPSILON_START  = 0.20
    EPSILON_END    = 0.01
    EPSILON_DECAY  = 0.9940

    # ---- Run length ----
    NUM_EPISODES = 500

    # ---- Storage (isolated from base training) ----
    MODEL_SAVE_DIR = "model_finetune"
    LOG_DIR        = "logs/finetune"

    # ---- Replay buffer ----
    MEMORY_SIZE     = 20_000
    MIN_MEMORY_SIZE = 4_000

    # ---- Checkpointing ----
    SAVE_FREQUENCY = 50

    # ---- Reward parameters (defaults; overridden per-episode by MixedTrafficEnv) ----
    REWARD_PHASE_CHANGE = -1.0
    SPEED_THRESHOLD     = 2.0


# ============================================================================
# Base-model checkpoint constants
# ============================================================================

_BASE_CHECKPOINT_DIR = "model/episode_1000"
_BASE_EPISODE        = 1000
_FINETUNE_EMB_FNAME  = "finetune_ready_embedding.pth"


# ============================================================================
# FineTuneTrainer
# ============================================================================

class FineTuneTrainer(Trainer):
    """
    Training manager for the 3-in-1 fine-tuning phase.

    Inherits all infrastructure from ``Trainer`` (training loop,
    progress reporting, checkpointing, JSON logging) and overrides
    only the three initialisation methods that differ:

      _init_environment()         → MixedTrafficEnv  (82-D obs)
      _init_agent()               → ABSTLightAgent(obs_dim=82, lr=5e-5, ε=0.20)
      _init_memory()              → fresh ReplayBuffer  (N_agents, 82)

    Adds:
      _gateway_check()            → abort if finetune_ready_embedding.pth missing
      _assemble_pretrained_weights() → load surgically modified embedding +
                                       original GCN-MHA + Q-heads
      _resume_from_latest_checkpoint() → first run  → assemble base weights
                                          subsequent → resume fine-tuning ckpt
    """

    def __init__(
        self,
        config=FineTuneConfig,
        use_gui: Optional[bool] = None,
        num_episodes: Optional[int] = None,
        epsilon_override: Optional[float] = None,
    ):
        # Optional override used when resuming to force a specific epsilon value.
        self._epsilon_override = epsilon_override
        # NOTE: we intentionally do NOT pass route_file here because
        # MixedTrafficEnv manages route selection internally per episode.
        super().__init__(
            config=config,
            use_gui=use_gui,
            num_episodes=num_episodes,
            route_file=None,
        )

    # ------------------------------------------------------------------
    # Environment initialisation  (override)
    # ------------------------------------------------------------------

    def _init_environment(self) -> None:
        """
        Create a MixedTrafficEnv and resolve obs_dim / num_agents / num_actions.

        MixedTrafficEnv passes obs_dim=80 to its parent internally (to satisfy
        the stacking divisibility constraint) but exposes total_obs_dim=82.
        We set self.obs_dim = 82 here so that all downstream initialisation
        (agent, memory, obs shape checks) uses the correct 82-D dimension.
        """
        print("\nInitialising Mixed Traffic Environment (3-in-1 Domain Randomisation)...")

        reward_weights = {
            "throughput":       self.config.REWARD_THROUGHPUT_WEIGHT,
            "pressure":         self.config.REWARD_PRESSURE_WEIGHT,
            "insertion_penalty": self.config.REWARD_INSERTION_PENALTY_WEIGHT,
            "phase_change":     self.config.REWARD_PHASE_CHANGE,
        }

        self.env = MixedTrafficEnv(
            sumo_config=self.config.SUMO_CONFIG_PATH,
            use_gui=self.config.USE_GUI,
            num_actions=self.config.NUM_ACTIONS,
            obs_stack_size=int(getattr(self.config, "OBS_STACK_SIZE", 4)),
            yellow_duration=self.config.YELLOW_PHASE_DURATION,
            min_green_duration=self.config.MIN_GREEN_DURATION,
            reward_weights=reward_weights,
            max_steps=self.config.MAX_STEPS_PER_EPISODE,
            pressure_normalization_capacity=float(
                getattr(self.config, "PRESSURE_NORMALIZATION_CAPACITY", 50.0)
            ),
            insertion_penalty_clip=float(
                getattr(self.config, "INSERTION_PENALTY_CLIP", -30.0)
            ),
            obs_queue_normalization=float(
                getattr(self.config, "OBS_QUEUE_NORMALIZATION", 50.0)
            ),
            obs_waiting_time_normalization=float(
                getattr(self.config, "OBS_WAITING_TIME_NORMALIZATION", 300.0)
            ),
            scenario_probs={"sunny": 0.50, "rainy": 0.30, "accident": 0.20},
            dynamic_accident_activation=True,
            accident_vehicle_id="accident_veh",
        )

        # Detect effective action space and agent count from SUMO
        self.num_actions = self.env.detect_action_space()
        detected_agents  = self.env.detect_num_agents()

        self.config.NUM_AGENTS = detected_agents
        self.num_agents        = detected_agents

        # IMPORTANT: use the 82-D total_obs_dim, NOT the 80-D internal obs_dim
        self.obs_dim = self.env.total_obs_dim  # 82

        print(f"  SUMO Config        : {self.config.SUMO_CONFIG_PATH}")
        print(f"  Obs dim (agent)    : {self.obs_dim}  (80D stacked + 2D context tag)")
        print(f"  Num agents         : {self.num_agents}")
        print(f"  Effective actions  : {self.num_actions}")
        print(
            f"  Scenario probs     : sunny=50%  rainy=30%  accident=20%"
        )
        print("  Dynamic accident   : enabled (accident_veh stop-triggered)")

    # ------------------------------------------------------------------
    # Agent initialisation  (override)
    # ------------------------------------------------------------------

    def _init_agent(self) -> None:
        """
        Initialise ABSTLightAgent with obs_dim=82 and fine-tuning hyperparameters.

        The pre-trained weights are NOT loaded here; that happens later in
        ``_assemble_pretrained_weights()`` which is called from
        ``_resume_from_latest_checkpoint()`` on the first run.
        """
        print("\nInitialising ABSTLight Agent (obs_dim=82, lr=5e-5, ε₀=0.20)...")

        adj_matrix = self.env.get_adjacency_matrix().astype(np.float32)
        if adj_matrix.shape != (self.num_agents, self.num_agents):
            raise ValueError(
                f"Adjacency shape mismatch: got {adj_matrix.shape}, "
                f"expected ({self.num_agents}, {self.num_agents})."
            )

        self.agent = ABSTLightAgent(
            obs_dim=self.obs_dim,                   # 82
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
            learning_rate=float(self.config.LEARNING_RATE),  # 5e-5
            gamma=float(self.config.GAMMA),
            error_threshold=float(self.config.ERROR_THRESHOLD),
            reward_scale_divisor=float(self.config.REWARD_SCALE_DIVISOR),
            use_dynamic_error_threshold=bool(self.config.USE_DYNAMIC_ERROR_THRESHOLD),
            relative_error_threshold_ratio=float(
                self.config.RELATIVE_ERROR_THRESHOLD_RATIO
            ),
            min_target_scale_for_threshold=float(
                self.config.MIN_TARGET_SCALE_FOR_THRESHOLD
            ),
            max_resampling_attempts=int(self.config.MAX_RESAMPLING_ATTEMPTS),
            epsilon_start=float(self.config.EPSILON_START),   # 0.20
            epsilon_end=float(self.config.EPSILON_END),       # 0.01
            epsilon_decay=float(self.config.EPSILON_DECAY),   # 0.9940
            device=self.device,
        )

        print(f"  obs_dim         : {self.obs_dim}")
        print(f"  learning_rate   : {self.config.LEARNING_RATE}")
        print(f"  epsilon         : {self.config.EPSILON_START} → {self.config.EPSILON_END}"
              f"  (decay={self.config.EPSILON_DECAY})"
        )

    # ------------------------------------------------------------------
    # Memory initialisation  (override)
    # ------------------------------------------------------------------

    def _init_memory(self) -> None:
        """
        Create a completely fresh 82-D replay buffer.

        No 80-D experiences from base training are reused; the old buffer is
        dimensionally incompatible and must NOT be loaded.
        """
        print("\nInitialising Fresh Replay Buffer (82-D, capacity=20 000)...")

        state_shape = (self.num_agents, self.obs_dim)   # (N_agents, 82)

        self.memory = ReplayBuffer(
            capacity=self.config.MEMORY_SIZE,
            state_shape=state_shape,
            device=self.device,
        )

        print(f"  State shape     : {state_shape}")
        print(f"  Capacity        : {self.config.MEMORY_SIZE}")
        print(f"  Min for training: {self.config.MIN_MEMORY_SIZE}")
        print("  [NOTE] No 80-D experiences from base training are imported.")

    # ------------------------------------------------------------------
    # Route scheduling  (no-op override)
    # ------------------------------------------------------------------

    def _apply_episode_route(self, episode: int) -> None:
        """
        Route selection is handled internally by MixedTrafficEnv.reset().

        The parent's implementation would overwrite self.env.route_file with
        the Sunny training schedule, which would then be immediately overridden
        by _sample_and_apply_scenario() anyway.  We suppress the parent's
        call entirely to avoid spurious FileNotFoundError checks and confusing
        log messages.
        """
        # Intentionally a no-op: MixedTrafficEnv._sample_and_apply_scenario()
        # sets self.env.route_file before each SUMO launch inside reset().
        pass

    # ------------------------------------------------------------------
    # Gateway check
    # ------------------------------------------------------------------

    def _gateway_check(self) -> None:
        """
        Abort with a clear message if the surgical embedding checkpoint is missing.

        This is the mandatory prerequisite for starting fine-tuning.
        """
        base_dir  = Path(_BASE_CHECKPOINT_DIR)
        emb_file  = base_dir / _FINETUNE_EMB_FNAME
        gcn_file  = base_dir / f"gcn_mha_episode_{_BASE_EPISODE}.pth"
        state_file = base_dir / f"agent_state_episode_{_BASE_EPISODE}.pth"

        errors: List[str] = []

        if not emb_file.exists():
            errors.append(
                f"  [MISSING] {emb_file}\n"
                "            Run:  python weight_converter.py"
            )
        if not gcn_file.exists():
            errors.append(
                f"  [MISSING] {gcn_file}\n"
                "            Train the base model to episode 1000 first."
            )
        if not state_file.exists():
            errors.append(
                f"  [MISSING] {state_file}\n"
                "            Train the base model to episode 1000 first."
            )

        num_agents  = self.num_agents
        num_networks = self.config.N_NETWORKS
        for i in range(num_agents):
            for k in range(num_networks):
                q_file = base_dir / f"q_head_agent{i}_net{k}_episode_{_BASE_EPISODE}.pth"
                if not q_file.exists():
                    errors.append(f"  [MISSING] {q_file}")

        if errors:
            print("\n" + "=" * 65)
            print("  GATEWAY CHECK FAILED — Fine-tuning prerequisites not met")
            print("=" * 65)
            for msg in errors:
                print(msg)
            print()
            print("  Resolution steps:")
            print(f"  1. Train base model:   python train.py   (to episode {_BASE_EPISODE})")
            print(f"  2. Run surgery:        python weight_converter.py")
            print(f"  3. Retry fine-tuning:  python train_finetune.py")
            print("=" * 65)
            sys.exit(1)

        print("[GATEWAY] All base-model prerequisites found. Proceeding...")

    # ------------------------------------------------------------------
    # Weight assembly  (first-run only)
    # ------------------------------------------------------------------

    def _assemble_pretrained_weights(self) -> None:
        """
        Load the surgically modified embedding and the original base-model
        GCN-MHA backbone and Q-heads into the freshly constructed 82-D agent.

        Call order:
            1. Load finetune_ready_embedding.pth  → embedding layer (82-D)
            2. Load gcn_mha_episode_1000.pth     → GCN-MHA stack (unchanged)
            3. Load q_head_agent{i}_net{k}_episode_1000.pth for every head

        After this call the network outputs EXACTLY the same decisions as the
        1000-episode base model for any Sunny input (context tag = [0, 0]),
        because the zero-padded columns contribute nothing to the forward pass.
        """
        base_dir = Path(_BASE_CHECKPOINT_DIR)
        device   = self.device

        print("\nAssembling pre-trained weights into 82-D agent...")

        # ----------------------------------------------------------------
        # 1. Surgically modified embedding (80D → 82D, new cols = 0)
        # ----------------------------------------------------------------
        emb_path = base_dir / _FINETUNE_EMB_FNAME
        emb_state = torch.load(str(emb_path), map_location=device)

        # Validate shape before loading
        expected_shape = (self.config.EMBED_DIM, 82)
        actual_shape   = tuple(emb_state["embedding.weight"].shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Unexpected embedding.weight shape in {emb_path}: "
                f"got {actual_shape}, expected {expected_shape}.\n"
                "Re-run weight_converter.py to regenerate the file."
            )

        self.agent.embedding.load_state_dict(emb_state, strict=True)
        print(f"  [OK] Embedding loaded  :  {emb_path}")
        print(f"       weight shape       :  {list(emb_state['embedding.weight'].shape)}")

        # ----------------------------------------------------------------
        # 2. GCN-MHA backbone  (architecture unchanged, weights intact)
        # ----------------------------------------------------------------
        gcn_path  = base_dir / f"gcn_mha_episode_{_BASE_EPISODE}.pth"
        gcn_state = torch.load(str(gcn_path), map_location=device)
        self.agent.gcn_mha.load_state_dict(gcn_state, strict=True)
        print(f"  [OK] GCN-MHA loaded    :  {gcn_path}")

        # ----------------------------------------------------------------
        # 3. Q-head ensemble  (architecture unchanged, weights intact)
        # ----------------------------------------------------------------
        loaded_heads = 0
        for i, agent_heads in enumerate(self.agent.q_heads):
            for k, head in enumerate(agent_heads):
                q_path = base_dir / f"q_head_agent{i}_net{k}_episode_{_BASE_EPISODE}.pth"
                head.load(str(q_path))
                loaded_heads += 1

        print(
            f"  [OK] Q-heads loaded    :  "
            f"{loaded_heads} heads  "
            f"({self.num_agents} agents × {self.config.N_NETWORKS} networks)"
        )

        print()
        print("  Weight assembly complete.")
        print("  Network output is identical to the 1000-episode base model")
        print("  for any input with context tag [0, 0]  (Sunny).")

    # ------------------------------------------------------------------
    # Checkpoint resume logic  (override)
    # ------------------------------------------------------------------

    def _resume_from_latest_checkpoint(self) -> int:
        """
        Decide whether this is a first run or a resumed fine-tuning run.

        First run  (model_finetune/ is empty / nonexistent):
            → _gateway_check() verifies all base-model prerequisites.
            → _assemble_pretrained_weights() loads the surgical embedding
              plus the original GCN-MHA and Q-heads.
            → Returns 1  (start fine-tuning from episode 1).

        Resumed run  (model_finetune/episode_N/ exists):
            → agent.load() restores the fine-tuning checkpoint.
            → Returns N + 1  (continue from where we left off).
            → _assemble_pretrained_weights() is NOT called to avoid
              overwriting the already fine-tuned weights.
        """
        checkpoint_dir, checkpoint_episode = find_latest_checkpoint(
            self.config.MODEL_SAVE_DIR
        )

        if checkpoint_dir is not None and checkpoint_episode is not None:
            # ---- Resumed fine-tuning run ----
            print(
                f"\nResuming fine-tuning from checkpoint: {checkpoint_dir}"
            )
            self.agent.load(checkpoint_dir, episode=checkpoint_episode)
            if self._epsilon_override is not None:
                self.agent.epsilon = float(self._epsilon_override)
                print(
                    f"Epsilon overridden to {self._epsilon_override} via --epsilon flag."
                )
            next_episode = checkpoint_episode + 1
            print(f"Continuing from episode {next_episode}.")
            return next_episode

        # ---- First run: perform full weight assembly ----
        print("\nNo existing fine-tuning checkpoint found — starting fresh.")
        self._gateway_check()
        self._assemble_pretrained_weights()
        print("\nFine-tuning initialisation complete. Starting from episode 1.")
        return 1

    # ------------------------------------------------------------------
    # Observation shape adapter  (override)
    # ------------------------------------------------------------------

    def _to_flat_multi_agent_obs(self, state: np.ndarray) -> np.ndarray:
        """
        Validate and return the (N_agents, 82) observation array.

        Overrides the parent's version to handle self.obs_dim = 82
        (the parent was written assuming obs_dim equals the stacking dim).
        """
        arr = np.asarray(state, dtype=np.float32)

        if arr.ndim == 2 and arr.shape == (self.num_agents, self.obs_dim):
            return arr

        if arr.ndim == 1 and self.num_agents == 1 and arr.shape[0] == self.obs_dim:
            return arr.reshape(1, self.obs_dim)

        raise ValueError(
            f"Observation shape mismatch: got {arr.shape}, "
            f"expected ({self.num_agents}, {self.obs_dim})."
        )

    # ------------------------------------------------------------------
    # Training-loop banner  (override to clarify mode)
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Execute the 500-episode fine-tuning loop.

        Adds a clear header banner and delegates to the inherited Trainer.train()
        which handles episode looping, checkpointing, and JSON logging.
        """
        print("\n" + "=" * 65)
        print("  ABSTLight  —  3-in-1 Fine-Tuning Phase")
        print("=" * 65)
        print(f"  Base model       : {_BASE_CHECKPOINT_DIR}/")
        print(f"  Fine-tune output : {self.config.MODEL_SAVE_DIR}/")
        print(f"  Episodes         : {self.config.NUM_EPISODES}")
        print(f"  Learning rate    : {self.config.LEARNING_RATE}")
        print(f"  Epsilon schedule : {self.config.EPSILON_START} → "
              f"{self.config.EPSILON_END}  (decay={self.config.EPSILON_DECAY})")
        print(f"  Buffer capacity  : {self.config.MEMORY_SIZE}  (fresh, 82-D)")
        print("=" * 65)

        super().train()


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ABSTLight 3-in-1 Fine-Tuning Pipeline\n"
            "\n"
            "Prerequisites:\n"
            f"  1. Base model checkpoint in {_BASE_CHECKPOINT_DIR}/\n"
            f"  2. Surgical embedding:  python weight_converter.py\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch SUMO with the graphical interface.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override NUMBER of fine-tuning episodes (default: 500).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Override starting epsilon when resuming a checkpoint (e.g. 0.2).",
    )
    args = parser.parse_args()

    trainer = FineTuneTrainer(
        config=FineTuneConfig,
        use_gui=args.gui,
        num_episodes=args.episodes,
        epsilon_override=args.epsilon,
    )
    trainer.train()


if __name__ == "__main__":
    main()
