"""
ABSTLight Agent  —  Adaptive Behavioral Spatial-Temporal Light
==============================================================

This module is the top-level integration layer that stitches together the
three ABSTLight processing layers and preserves the full RELight ensemble
algorithm underneath them.

Pipeline overview
-----------------
                ┌─────────────────────────────────────────────────────────┐
                │ Shared front-end (one instance, updated jointly)        │
                │                                                         │
  o_i^t  ──►  [ Layer 1: ObservationEmbedding ]  ──►  h_i               │
  (all N       [ Layer 2: GCNMHAStack + adj     ]  ──►  h_{m_i}^L        │
  agents)       └─────────────────────────────────────────────────────────┘
                     │  h_{m_i}^L : (B, embed_dim) per agent i
                     ▼
             ┌ Agent 0 ─── [ Q-head 0 ] ──┐
             │              [ Q-head 1 ]   │  RELight ensemble
             │              ...            │  (N heads per agent)
             │              [ Q-head N-1 ] │
             └─────────────────────────────┘
             ┌ Agent 1 ─── [ Q-head 0 ] ──┐
             │              ...            │
             └─────────────────────────────┘
             ...
             ┌ Agent K ─── [ Q-head 0 ] ──┐
             │              ...            │
             └─────────────────────────────┘

Training uses ABSTLightEnsembleUpdater which:
  1. Runs the shared front-end ONCE on the batch of all-agent observations.
  2. For each agent i and each Q-head k, applies the RELight
     error-threshold–based M-subset resampling algorithm to choose a
     conservative target from the ensemble, then computes MSE loss.
  3. Accumulates all losses (from all agents and all Q-heads) and performs
     a SINGLE backward pass so that the shared front-end receives gradient
     contributions from every Q-head update simultaneously.

Single-intersection compatibility
----------------------------------
When num_agents=1 and adj = [[1]], the GCN-MHA layer degenerates to pure
self-attention (each node attends only to itself) and the output equals
the embedding output after the residual + LayerNorm step.  This recovers a
behaviour equivalent to a standard DQN embedding, so the ABSTLightAgent
can be used as a drop-in upgrade to a single-intersection RELightAgent as
well.

Public API (mirrors RELightAgent)
----------------------------------
  select_action(all_obs, agent_idx, mode)  → int
  select_all_actions(all_obs, mode)        → List[int]
  update(all_states, actions, rewards, all_next_states, dones) → stats
  decay_epsilon() / set_epsilon()
  train_mode() / eval_mode()
  save(directory, episode) / load(directory, episode)
  get_statistics() → dict
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from model_components.observation_embedding import ObservationEmbedding
from model_components.gcn_mha import GCNMHAStack
from model_components.q_value_head import QValueHead


# ============================================================================
# ABSTLight Ensemble Updater  (RELight algorithm over flat spatial features)
# ============================================================================

class ABSTLightEnsembleUpdater:
    """
    Ensemble update module for ABSTLight.

    Adapts the RELight "data-repeated-sampling" algorithm (from
    EnsembleUpdater) to work with the shared spatial-temporal front-end.

    Key differences from EnsembleUpdater
    -------------------------------------
    - Input to the Q-heads is *not* a raw image or flat obs vector; it is the
      spatial feature h_{m_i}^L produced by ObservationEmbedding + GCNMHAStack.
    - The shared front-end (embedding + gcn_mha) participates in backprop and
      its parameters are updated via a single joint backward pass that
      accumulates gradient contributions from every (agent, Q-head) pair.
    - The per-agent, per-Q-head optimizer handles only that head's parameters.

    RELight algorithm (preserved)
    ------------------------------
    For each agent i and Q-head k in the ensemble:
      1. Predict Q-values: q_pred = QValueHead_k( h_{m_i}^L )
      2. Sample an M-subset of Q-heads (RELight random subset sampling).
      3. Compute target: reward + γ · min_{j ∈ subset}( max_a Q_j(h_next) )
      4. Error = MSE(q_pred, target).
      5. If error > threshold: resample subset (up to max_attempts).
      6. Accumulate MSE loss for the accepted (q_pred, target) pair.
    After iterating all (i, k) pairs:
      7. Single backward pass over the total accumulated loss.
      8. Shared front-end optimizer step.
      9. Per–(agent, Q-head) optimizer steps.

    Args:
        embedding              : Shared ObservationEmbedding module.
        gcn_mha                : Shared GCNMHAStack module.
        q_heads                : nn.ModuleList[ nn.ModuleList[QValueHead] ]
                                 Indexed as q_heads[agent_idx][network_idx].
        num_agents       (int) : Number of intersections N.
        N                (int) : Number of Q-heads per agent (ensemble size).
        M                (int) : Random subset size for target Q-value.
        learning_rate  (float) : Shared LR for all optimizers.
        gamma          (float) : Discount factor γ.
        error_threshold(float) : Maximum acceptable MSE for subset acceptance.
        reward_scale_divisor (float): Reward divisor for Bellman targets.
        use_dynamic_error_threshold (bool): Enable relative thresholding.
        relative_error_threshold_ratio (float): Relative threshold ratio.
        min_target_scale_for_threshold (float): Floor for relative scaling.
        max_resampling_attempts(int): Max subset resampling iterations.
        device         (str)  : Torch device string.
    """

    def __init__(
        self,
        embedding: ObservationEmbedding,
        gcn_mha: GCNMHAStack,
        q_heads: nn.ModuleList,
        num_agents: int,
        N: int,
        M: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.95,
        error_threshold: float = 150.0,
        reward_scale_divisor: float = 1000.0,
        use_dynamic_error_threshold: bool = True,
        relative_error_threshold_ratio: float = 0.05,
        min_target_scale_for_threshold: float = 1.0,
        max_resampling_attempts: int = 5,
        device: str = "cpu",
    ):
        self.embedding = embedding
        self.gcn_mha = gcn_mha
        self.q_heads = q_heads
        self.num_agents = num_agents
        self.N = N
        self.M = M
        self.gamma = gamma
        self.error_threshold = error_threshold
        self.reward_scale_divisor = float(reward_scale_divisor)
        self.use_dynamic_error_threshold = bool(use_dynamic_error_threshold)
        self.relative_error_threshold_ratio = float(relative_error_threshold_ratio)
        self.min_target_scale_for_threshold = float(min_target_scale_for_threshold)
        self.max_resampling_attempts = max_resampling_attempts
        self.device = device

        if self.reward_scale_divisor <= 0.0:
            raise ValueError("reward_scale_divisor must be > 0")
        if self.relative_error_threshold_ratio < 0.0:
            raise ValueError("relative_error_threshold_ratio must be >= 0")
        if self.min_target_scale_for_threshold <= 0.0:
            raise ValueError("min_target_scale_for_threshold must be > 0")

        # ------------------------------------------------------------------
        # Optimizers
        # ------------------------------------------------------------------
        # Shared front-end optimizer — covers ObservationEmbedding and GCNMHAStack.
        # The front-end receives gradients accumulated from ALL (agent, head) losses.
        frontend_params = (
            list(embedding.parameters()) + list(gcn_mha.parameters())
        )
        self.frontend_optimizer = optim.Adam(frontend_params, lr=learning_rate)

        # Per-agent × per-Q-head optimizers — each covers one QValueHead only.
        # head_optimizers[i][k] → optimizer for q_heads[i][k]
        self.head_optimizers: List[List[optim.Optimizer]] = [
            [
                optim.Adam(q_heads[i][k].parameters(), lr=learning_rate)
                for k in range(N)
            ]
            for i in range(num_agents)
        ]

        # Huber loss is more robust than MSE for occasional large TD errors.
        self.criterion = nn.SmoothL1Loss()

        # Statistics (mirrors EnsembleUpdater.update_stats)
        self.update_stats = {
            "total_updates": 0,
            "resampling_count": 0,
            "avg_resampling_per_update": 0.0,
            "rejected_subsets": 0,
            "skipped_updates": 0,
            "threshold_evaluations": 0,
            "avg_effective_threshold": 0.0,
            "avg_target_abs_mean": 0.0,
        }

    # ------------------------------------------------------------------
    # RELight subset sampling helpers
    # ------------------------------------------------------------------

    def _sample_subset(
        self, M: int, exclude: Optional[List[int]] = None
    ) -> List[int]:
        """
        Randomly sample M Q-head indices from [0, N) without replacement.

        Args:
            M       : Number of indices to sample.
            exclude : Indices to exclude from the pool (optional).

        Returns:
            List of M randomly drawn head indices.
        """
        pool = list(range(self.N))
        if exclude:
            pool = [i for i in pool if i not in exclude]
        M = min(M, len(pool))
        return np.random.choice(pool, size=M, replace=False).tolist()

    def _compute_target_q(
        self,
        subset_indices: List[int],
        agent_idx: int,
        H_L_next: torch.Tensor,    # (B, N_agents, embed_dim)  — detached
        rewards: torch.Tensor,      # (B,)
        dones: torch.Tensor,        # (B,)
    ) -> torch.Tensor:
        """
        Compute the RELight conservative target Q-value for one agent.

        Target = reward + γ · min_{k ∈ subset} max_a Q_k( h_{m_i,next}^L )

        The minimum-over-subset is the key RELight innovation that reduces
        overestimation bias without a separate target network.

        Args:
            subset_indices : M Q-head indices used as the target ensemble.
            agent_idx      : Which intersection to compute targets for.
            H_L_next       : Spatial features for next states (detached).
                             Shape (B, N_agents, embed_dim).
            rewards        : Per-step rewards for agent_idx, shape (B,).
            dones          : Episode termination flags, shape (B,).

        Returns:
            target_q : Shape (B,), always detached (no gradient).

        Tensor dimension trace:
            H_L_next[:, agent_idx, :] : (B, embed_dim)  ← feature for agent i
            Q_k(h_next)               : (B, num_actions) per head k
            max_a Q_k                 : (B,)             per head k
            min over subset           : (B,)             conservative estimate
            target_q                  : (B,)
        """
        h_next = H_L_next[:, agent_idx, :]   # (B, embed_dim)

        with torch.no_grad():
            # Max Q-value at next state for each head in the subset: (B,) each
            max_q_per_head = [
                self.q_heads[agent_idx][k](h_next).max(dim=1).values
                for k in subset_indices
            ]
            # Stack → (M, B), then take element-wise minimum across heads → (B,)
            q_min = torch.stack(max_q_per_head, dim=0).min(dim=0).values   # (B,)

        # Bellman target: only bootstrap from non-terminal transitions
        scaled_rewards = rewards / self.reward_scale_divisor
        target_q = scaled_rewards + self.gamma * q_min * (1.0 - dones)
        return target_q.detach()  # no gradient through target

    def _effective_error_threshold(self, target_q: torch.Tensor) -> Tuple[float, float]:
        """Compute absolute+relative gate threshold based on target scale."""
        target_abs_mean = float(target_q.detach().abs().mean().item())
        # Scale base threshold to match typical target magnitude (accounting for reward_scale_divisor)
        scaled_base_threshold = self.error_threshold / self.reward_scale_divisor
        if self.use_dynamic_error_threshold:
            relative_component = self.relative_error_threshold_ratio * max(
                target_abs_mean,
                self.min_target_scale_for_threshold,
            )
            threshold = max(scaled_base_threshold, relative_component)
        else:
            threshold = scaled_base_threshold
        return float(threshold), target_abs_mean

    # ------------------------------------------------------------------
    # Main update entry-point
    # ------------------------------------------------------------------

    def update_all(
        self,
        all_states: torch.Tensor,       # (B, N_agents, obs_dim)
        actions: torch.Tensor,          # (B, N_agents)   int64
        rewards: torch.Tensor,          # (B, N_agents)   float32
        all_next_states: torch.Tensor,  # (B, N_agents, obs_dim)
        dones: torch.Tensor,            # (B,)            float32  {0, 1}
        adj: torch.Tensor,              # (N_agents, N_agents)
        M: int,
    ) -> List[List[Dict]]:
        """
        End-to-end ABSTLight ensemble update for one training batch.

        Step 1  —  Shared front-end on current states (retains grad).
        Step 2  —  Shared front-end on next states (no grad; for targets).
        Step 3  —  For each (agent, Q-head): compute RELight loss.
        Step 4  —  Single joint backward pass; step all optimizers.

        Args:
            all_states      : Shape (B, N_agents, obs_dim).
            actions         : Shape (B, N_agents), dtype int64.
            rewards         : Shape (B, N_agents), dtype float32.
            all_next_states : Shape (B, N_agents, obs_dim).
            dones           : Shape (B,), dtype float32.
            adj             : Shape (N_agents, N_agents).
            M               : Subset size for RELight target computation.

        Returns:
            results[i][k] : dict with 'loss' and 'resampling_attempts' for
                            agent i, Q-head k.

        Tensor dimension trace (key shapes)
        ------------------------------------
            all_states      : (B, N, obs_dim)
            H               : (B, N, embed_dim)        ← after ObsEmbed
            H_L             : (B, N, embed_dim)        ← after GCN-MHA (w/ grad)
            H_L_next        : (B, N, embed_dim)        ← after GCN-MHA (no grad)
            h_i             : (B, embed_dim)           ← H_L[:, i, :]
            q_all           : (B, num_actions)         ← QValueHead_k(h_i)
            q_pred          : (B,)                     ← gathered at taken action
            target_q        : (B,)                     ← RELight conservative target
            loss            : scalar
            total_loss      : scalar                   ← sum over all (i, k)
        """
        # ---- Zero all gradients before building the computation graph ----
        self.frontend_optimizer.zero_grad()
        for i in range(self.num_agents):
            for k in range(self.N):
                self.head_optimizers[i][k].zero_grad()

        # ====================================================================
        # Step 1: Shared front-end on CURRENT states  (gradient retained)
        # ====================================================================
        # ObservationEmbedding: (B, N, obs_dim) → (B, N, embed_dim)
        H = self.embedding(all_states)
        # GCNMHAStack: (B, N, embed_dim) → (B, N, embed_dim)
        H_L = self.gcn_mha(H, adj)                           # (B, N, embed_dim)

        # ====================================================================
        # Step 2: Shared front-end on NEXT states  (no gradient needed)
        # ====================================================================
        with torch.no_grad():
            H_next = self.embedding(all_next_states)          # (B, N, embed_dim)
            H_L_next = self.gcn_mha(H_next, adj)              # (B, N, embed_dim)

        # ====================================================================
        # Step 3: RELight ensemble loss for every (agent i, Q-head k) pair
        # ====================================================================
        losses: List[torch.Tensor] = []
        results: List[List[Dict]] = [
            [None] * self.N for _ in range(self.num_agents)
        ]

        total_resamples_this_update = 0

        for i in range(self.num_agents):
            # Select the feature slice for agent i
            # h_i shape: (B, embed_dim) — view into H_L with gradient
            h_i = H_L[:, i, :]                                # (B, embed_dim)

            # Agent i's actions and rewards for this batch
            agent_actions = actions[:, i]                     # (B,)  int64
            agent_rewards = rewards[:, i]                     # (B,)  float32

            for k in range(self.N):
                # ---- Q prediction from Q-head k ----------------------------
                # q_all  : (B, num_actions)
                q_all = self.q_heads[i][k](h_i)
                # q_pred : (B,)  — Q-value for the *actually taken* action
                q_pred = q_all.gather(
                    1, agent_actions.unsqueeze(1)
                ).squeeze(1)                                   # (B,)

                # ---- RELight: error-threshold–based subset resampling ------
                accepted_subset: Optional[List[int]] = None
                final_error = float("inf")
                step_resamples = 0
                effective_threshold = self.error_threshold
                target_abs_mean = 0.0

                for attempt in range(self.max_resampling_attempts):
                    subset = self._sample_subset(M, exclude=[k])
                    q_target = self._compute_target_q(
                        subset, i, H_L_next, agent_rewards, dones
                    )
                    # Evaluate error WITHOUT recording gradients for the check
                    candidate_error = self.criterion(q_pred.detach(), q_target).item()
                    effective_threshold, target_abs_mean = self._effective_error_threshold(q_target)
                    self.update_stats["threshold_evaluations"] += 1
                    n_evals = self.update_stats["threshold_evaluations"]
                    self.update_stats["avg_effective_threshold"] += (
                        effective_threshold - self.update_stats["avg_effective_threshold"]
                    ) / n_evals
                    self.update_stats["avg_target_abs_mean"] += (
                        target_abs_mean - self.update_stats["avg_target_abs_mean"]
                    ) / n_evals
                    final_error = candidate_error

                    if candidate_error <= effective_threshold:
                        accepted_subset = subset
                        break
                    else:
                        self.update_stats["rejected_subsets"] += 1
                        step_resamples += 1

                total_resamples_this_update += step_resamples
                self.update_stats["resampling_count"] += step_resamples

                # If no subset satisfied threshold, skip this head update safely.
                if accepted_subset is None:
                    self.update_stats["skipped_updates"] += 1
                    results[i][k] = {
                        "loss": None,
                        "resampling_attempts": step_resamples,
                        "final_error": final_error,
                        "effective_threshold": effective_threshold,
                        "target_abs_mean": target_abs_mean,
                        "skipped": True,
                    }
                    continue

                # ---- Compute final loss with the accepted subset -----------
                q_target_final = self._compute_target_q(
                    accepted_subset, i, H_L_next, agent_rewards, dones
                )
                loss = self.criterion(q_pred, q_target_final)   # scalar
                losses.append(loss)

                results[i][k] = {
                    "loss": loss.item(),
                    "resampling_attempts": step_resamples,
                    "final_error": final_error,
                    "effective_threshold": effective_threshold,
                    "target_abs_mean": target_abs_mean,
                    "skipped": False,
                }

        # ====================================================================
        # Step 4: Single joint backward pass & optimizer steps
        # ====================================================================
        if losses:
            # Sum all (agent, Q-head) losses into a single scalar.
            # The shared front-end receives gradients accumulated from every
            # loss term; each Q-head's gradient flows only through its own fc1/fc2.
            total_loss = torch.stack(losses).sum()
            total_loss.backward()

            # Update shared front-end parameters (embedding + gcn_mha)
            self.frontend_optimizer.step()

            # Update each Q-head's parameters independently
            for i in range(self.num_agents):
                for k in range(self.N):
                    self.head_optimizers[i][k].step()

        # Update running statistics
        self.update_stats["total_updates"] += 1
        n_updates = self.update_stats["total_updates"]
        self.update_stats["avg_resampling_per_update"] = (
            self.update_stats["resampling_count"] / n_updates
        )

        return results

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        return dict(self.update_stats)


# ============================================================================
# ABSTLight Agent
# ============================================================================

class ABSTLightAgent:
    """
    ABSTLight Agent — multi-intersection traffic controller.

    Manages the full three-layer ABSTLight pipeline:
      Layer 1: ObservationEmbedding      (shared across all agents)
      Layer 2: GCNMHAStack               (shared across all agents)
      Layer 3: RELight Q-head ensemble   (N independent heads *per agent*)

    Data expected from the environment
    ------------------------------------
    Unlike the original RELightAgent which receives a single (C,H,W) image
    observation, ABSTLightAgent expects a flat observation vector per
    intersection:

      all_obs  shape: (N_agents, obs_dim)   — inference-time input
      states   shape: (B, N_agents, obs_dim) — training batch from replay buffer
      actions  shape: (B, N_agents)          — int64, per-agent actions
      rewards  shape: (B, N_agents)          — float32, per-agent rewards
      dones    shape: (B,)                   — bool/float32, episode end flag

    Where obs_dim is a flat vector of local traffic data (queue lengths, waiting
    times, current signal phase) for one intersection.  See Config.OBS_DIM.

    Args
    ----
    obs_dim          (int)          : Flat observation dimension per intersection.
    num_agents       (int)          : Number of intersections (graph nodes N).
    num_actions      (int)          : Number of traffic-signal phases.
    adj_matrix       (np.ndarray)   : (N, N) binary adjacency matrix.
                                      adj[i, j] = 1 iff i and j share a road link.
    embed_dim        (int)          : Hidden state dimensionality (Layer 1 output).
    num_heads        (int)          : MHA heads in GCNMHAStack.
    num_gcn_layers   (int)          : Depth L of GCNMHAStack.
    gcn_dropout      (float)        : Dropout used inside each GCNMHALayer.
    hidden_dim       (int)          : FC hidden layer width in QValueHead.
    N                (int)          : Ensemble size (Q-heads per agent).
    M                (int)          : Random subset size for RELight targets.
    learning_rate    (float)        : Adam learning rate for all optimizers.
    gamma            (float)        : Discount factor γ.
    error_threshold  (float)        : RELight subset rejection threshold.
    reward_scale_divisor (float)    : Reward divisor for Bellman targets.
    use_dynamic_error_threshold(bool): Enable relative thresholding.
    relative_error_threshold_ratio(float): Relative threshold ratio.
    min_target_scale_for_threshold(float): Floor for relative scaling.
    max_resampling_attempts (int)   : Max subset resampling iterations.
    epsilon_start    (float)        : Initial ε-greedy exploration rate.
    epsilon_end      (float)        : Minimum ε.
    epsilon_decay    (float)        : Multiplicative ε decay per episode.
    device           (str)          : PyTorch device string ('cpu' or 'cuda').
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        num_actions: int,
        adj_matrix: np.ndarray,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_gcn_layers: int = 2,
        gcn_dropout: float = 0.1,
        hidden_dim: int = 512,
        N: int = 10,
        M: int = 4,
        learning_rate: float = 1e-4,
        gamma: float = 0.95,
        error_threshold: float = 150.0,
        reward_scale_divisor: float = 1000.0,
        use_dynamic_error_threshold: bool = True,
        relative_error_threshold_ratio: float = 0.05,
        min_target_scale_for_threshold: float = 1.0,
        max_resampling_attempts: int = 5,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.N = N
        self.M = M
        self.device = device

        # ε-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Adjacency matrix stored as a non-trainable tensor buffer
        self.adj = torch.tensor(adj_matrix, dtype=torch.float32, device=device)

        # ==============================================================
        # Layer 1 — Observation Embedding  (shared across all agents)
        #   Input  : (..., obs_dim)
        #   Output : (..., embed_dim)
        # ==============================================================
        print("Initializing ABSTLight agent...")
        print(f"  Layer 1 | ObservationEmbedding : obs_dim={obs_dim} → embed_dim={embed_dim}")
        self.embedding = ObservationEmbedding(obs_dim, embed_dim).to(device)

        # ==============================================================
        # Layer 2 — GCN-MHA Stack  (shared across all agents)
        #   Input  : (..., N_agents, embed_dim) + adj (N_agents, N_agents)
        #   Output : (..., N_agents, embed_dim)
        # ==============================================================
        print(
            f"  Layer 2 | GCNMHAStack          : embed_dim={embed_dim}, "
            f"heads={num_heads}, layers={num_gcn_layers}"
        )
        self.gcn_mha = GCNMHAStack(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_gcn_layers,
            dropout=gcn_dropout,
        ).to(device)

        # ==============================================================
        # Layer 3 — RELight Q-head ensemble  (N heads per agent)
        #   Input  : (B, embed_dim) per agent
        #   Output : (B, num_actions) per agent
        # ==============================================================
        print(
            f"  Layer 3 | Q-head ensemble      : {num_agents} agents × "
            f"{N} heads, actions={num_actions}"
        )
        # q_heads[i] is an nn.ModuleList of N QValueHead objects for agent i.
        # The outer nn.ModuleList ensures all parameters are registered for
        # saving/loading with torch.save / state_dict().
        self.q_heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        QValueHead(embed_dim, num_actions, hidden_dim).to(device)
                        for _ in range(N)
                    ]
                )
                for _ in range(num_agents)
            ]
        )

        # ==============================================================
        # Ensemble Updater
        # ==============================================================
        self.updater = ABSTLightEnsembleUpdater(
            embedding=self.embedding,
            gcn_mha=self.gcn_mha,
            q_heads=self.q_heads,
            num_agents=num_agents,
            N=N,
            M=M,
            learning_rate=learning_rate,
            gamma=gamma,
            error_threshold=error_threshold,
            reward_scale_divisor=reward_scale_divisor,
            use_dynamic_error_threshold=use_dynamic_error_threshold,
            relative_error_threshold_ratio=relative_error_threshold_ratio,
            min_target_scale_for_threshold=min_target_scale_for_threshold,
            max_resampling_attempts=max_resampling_attempts,
            device=device,
        )

        self.training_step = 0
        self.episode_count = 0
        print("ABSTLight agent initialized.")

    # ------------------------------------------------------------------
    # Internal: shared front-end inference helper
    # ------------------------------------------------------------------

    def _run_frontend(
        self, all_obs: torch.Tensor, with_grad: bool = False
    ) -> torch.Tensor:
        """
        Run ObservationEmbedding + GCNMHAStack on all intersections' observations.

        Args:
            all_obs  : Observation tensor.
                       Inference : (N_agents, obs_dim)
                       Training  : (B, N_agents, obs_dim)
            with_grad: Whether to retain computation graph for backprop.

        Returns:
            H_L      : Spatial feature tensor, same leading shape as all_obs
                       but last dimension replaced by embed_dim.
                       Inference : (N_agents, embed_dim)
                       Training  : (B, N_agents, embed_dim)

        Tensor dimension trace:
            all_obs  : (N, obs_dim)   or  (B, N, obs_dim)
            H        : (N, embed_dim) or  (B, N, embed_dim)   ← ObsEmbed
            H_L      : (N, embed_dim) or  (B, N, embed_dim)   ← GCN-MHA
        """
        if with_grad:
            H = self.embedding(all_obs)       # (..., N, embed_dim)
            H_L = self.gcn_mha(H, self.adj)   # (..., N, embed_dim)
        else:
            with torch.no_grad():
                H = self.embedding(all_obs)
                H_L = self.gcn_mha(H, self.adj)
        return H_L

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        all_obs: torch.Tensor,
        agent_idx: int,
        mode: str = "voting",
    ) -> int:
        """
        Select action for one specific intersection using the full pipeline.

        Data flow:
          all_obs  →  [ObsEmbed + GCN-MHA]  →  H_L
          H_L[:, agent_idx, :]  →  N Q-heads (ensemble)  →  action

        Args:
            all_obs   : All intersections' observations, shape (N_agents, obs_dim).
            agent_idx : Index of the target intersection.
            mode      : Ensemble aggregation — 'voting', 'averaging', or
                        'random_network'.

        Returns:
            Selected action index (int).
        """
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.num_actions))

        all_obs = all_obs.to(self.device)
        H_L = self._run_frontend(all_obs, with_grad=False)  # (N_agents, embed_dim)
        h = H_L[agent_idx].unsqueeze(0)                      # (1, embed_dim)
        heads = self.q_heads[agent_idx]

        if mode == "voting":
            votes = np.zeros(self.num_actions)
            for head in heads:
                q = head(h)
                votes[q.argmax(dim=1).item()] += 1
            max_votes = votes.max()
            best = np.where(votes == max_votes)[0]
            return int(np.random.choice(best))

        elif mode == "averaging":
            q_sum = torch.zeros(1, self.num_actions, device=self.device)
            for head in heads:
                q_sum += head(h)
            return int((q_sum / self.N).argmax(dim=1).item())

        elif mode == "random_network":
            k = int(np.random.randint(0, self.N))
            return int(self.q_heads[agent_idx][k](h).argmax(dim=1).item())

        else:
            raise ValueError(f"Unknown action selection mode: {mode!r}")

    def select_all_actions(
        self,
        all_obs: torch.Tensor,
        mode: str = "voting",
    ) -> List[int]:
        """
        Select actions for ALL intersections in one efficient forward pass.

        The shared front-end runs ONCE and the per-agent ensemble heads are
        queried using the cached H_L slices.  This is significantly cheaper
        than N independent calls to select_action.

        Args:
            all_obs : All intersections' observations, shape (N_agents, obs_dim).
            mode    : Ensemble aggregation mode.

        Returns:
            actions : List of N_agents integer action indices.

        Tensor dimension trace:
            all_obs    : (N, obs_dim)
            H_L        : (N, embed_dim)    ← shared front-end
            h_i        : (1, embed_dim)    ← per-agent slice
            q_all_k    : (1, num_actions)  ← per-head Q-values
            votes/avg  : (num_actions,)    ← ensemble aggregation
        """
        all_obs = all_obs.to(self.device)
        H_L = self._run_frontend(all_obs, with_grad=False)  # (N_agents, embed_dim)

        actions: List[int] = []
        for i in range(self.num_agents):
            # Epsilon-greedy exploration per agent
            if np.random.random() < self.epsilon:
                actions.append(int(np.random.randint(0, self.num_actions)))
                continue

            h = H_L[i].unsqueeze(0)  # (1, embed_dim)
            heads = self.q_heads[i]

            if mode == "voting":
                votes = np.zeros(self.num_actions)
                for head in heads:
                    q = head(h)
                    votes[q.argmax(dim=1).item()] += 1
                max_v = votes.max()
                best = np.where(votes == max_v)[0]
                actions.append(int(np.random.choice(best)))

            elif mode == "averaging":
                q_sum = torch.zeros(1, self.num_actions, device=self.device)
                for head in heads:
                    q_sum += head(h)
                actions.append(int((q_sum / self.N).argmax(dim=1).item()))

            elif mode == "random_network":
                k = int(np.random.randint(0, self.N))
                actions.append(int(self.q_heads[i][k](h).argmax(dim=1).item()))

            else:
                raise ValueError(f"Unknown action selection mode: {mode!r}")

        return actions

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(
        self,
        all_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        all_next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> List[List[Dict]]:
        """
        One ABSTLight training step over a replay-buffer mini-batch.

        Delegates to ABSTLightEnsembleUpdater which runs the shared front-end
        and applies the RELight M-subset algorithm per (agent, Q-head) pair.

        Args:
            all_states      : (B, N_agents, obs_dim)   current observations
            actions         : (B, N_agents)            int64 actions taken
            rewards         : (B, N_agents)            float32 rewards received
            all_next_states : (B, N_agents, obs_dim)   next observations
            dones           : (B,)                     float32 {0, 1}

        Returns:
            results[i][k] — dict with 'loss', 'resampling_attempts', 'final_error'
                            for agent i, Q-head k.
        """
        all_states      = all_states.to(self.device)
        actions         = actions.to(self.device)
        rewards         = rewards.to(self.device)
        all_next_states = all_next_states.to(self.device)
        dones           = dones.to(self.device)

        results = self.updater.update_all(
            all_states,
            actions,
            rewards,
            all_next_states,
            dones,
            self.adj,
            self.M,
        )

        self.training_step += 1
        return results

    # ------------------------------------------------------------------
    # Exploration control
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """Multiply ε by the decay factor, clamped at ε_end."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def set_epsilon(self, epsilon: float):
        """Manually override exploration rate (clamped to [ε_end, ε_start])."""
        self.epsilon = max(
            self.epsilon_end, min(self.epsilon_start, float(epsilon))
        )

    def get_epsilon(self) -> float:
        return self.epsilon

    # ------------------------------------------------------------------
    # Training / eval mode
    # ------------------------------------------------------------------

    def train_mode(self):
        """Set all modules to training mode (enables dropout / batch-norm)."""
        self.embedding.train()
        self.gcn_mha.train()
        for agent_heads in self.q_heads:
            for head in agent_heads:
                head.train()

    def eval_mode(self):
        """Set all modules to evaluation mode (disables dropout)."""
        self.embedding.eval()
        self.gcn_mha.eval()
        for agent_heads in self.q_heads:
            for head in agent_heads:
                head.eval()

    # ------------------------------------------------------------------
    # Persistence  (mirrors RELightAgent API)
    # ------------------------------------------------------------------

    def save(self, directory: str, episode: Optional[int] = None):
        """
        Save all networks, the shared front-end, and agent state.

        Saved files inside `directory`:
          embedding[_episode_N].pth
          gcn_mha[_episode_N].pth
          q_head_{agent}_{net}[_episode_N].pth
          agent_state[_episode_N].pth

        Args:
            directory : Destination directory (created if absent).
            episode   : Optional episode number appended to filenames.
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        suffix = f"_episode_{episode}" if episode is not None else ""

        # Shared front-end
        torch.save(
            self.embedding.state_dict(),
            os.path.join(directory, f"embedding{suffix}.pth"),
        )
        torch.save(
            self.gcn_mha.state_dict(),
            os.path.join(directory, f"gcn_mha{suffix}.pth"),
        )

        # Per-agent Q-heads
        for i, agent_heads in enumerate(self.q_heads):
            for k, head in enumerate(agent_heads):
                fname = f"q_head_agent{i}_net{k}{suffix}.pth"
                head.save(os.path.join(directory, fname))

        # Agent meta-state
        torch.save(
            {
                "training_step": self.training_step,
                "episode_count": self.episode_count,
                "epsilon": self.epsilon,
                "updater_stats": self.updater.get_statistics(),
            },
            os.path.join(directory, f"agent_state{suffix}.pth"),
        )
        print(f"ABSTlight agent saved to {directory}")

    def load(self, directory: str, episode: Optional[int] = None):
        """
        Load all networks, the shared front-end, and agent state.

        Args:
            directory : Source directory.
            episode   : Optional episode number matching the saved filenames.
        """
        suffix = f"_episode_{episode}" if episode is not None else ""
        map_loc = self.device

        self.embedding.load_state_dict(
            torch.load(
                os.path.join(directory, f"embedding{suffix}.pth"),
                map_location=map_loc,
            )
        )
        self.gcn_mha.load_state_dict(
            torch.load(
                os.path.join(directory, f"gcn_mha{suffix}.pth"),
                map_location=map_loc,
            )
        )

        for i, agent_heads in enumerate(self.q_heads):
            for k, head in enumerate(agent_heads):
                fname = f"q_head_agent{i}_net{k}{suffix}.pth"
                head.load(os.path.join(directory, fname))

        state = torch.load(
            os.path.join(directory, f"agent_state{suffix}.pth"),
            map_location=map_loc,
        )
        self.training_step = state["training_step"]
        self.episode_count = state["episode_count"]
        self.epsilon = state["epsilon"]
        # Restore ensemble updater statistics if present
        if "updater_stats" in state:
            self.updater.update_stats = state["updater_stats"]
        print(f"ABSTLight agent loaded from {directory}")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        return {
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "epsilon": self.epsilon,
            "updater_stats": self.updater.get_statistics(),
        }
