"""
Q-Value Head  —  ABSTLight Layer 3 (per-intersection Q-network)
================================================================

Implements the Q-value prediction head described in the ABSTLight paper:

    q(o_i^t) = h_{m_i}^L · W_p + b_p

where:
  h_{m_i}^L  — the L-th layer spatial representation for intersection i,
                output of the GCN-MHA stack, shape (embed_dim,)
  W_p         — learnable projection weight, shape (embed_dim, num_actions)
                [stored transposed; equivalent to hidden_dim → num_actions]
  b_p         — learnable bias, shape (num_actions,)

In practice a single hidden layer (512 units with ReLU) is inserted between
the GCN-MHA output and the final Q-value projection, matching the architecture
of the original RELight BaseDQN fully-connected tail while removing the CNN
front-end (which is replaced by the spatial-temporal layers 1 and 2).

This module replaces BaseDQN for use within the ABSTLight ensemble:
  - BaseDQN  : Conv layers + FC512 + FC(num_actions) — processes raw images
  - QValueHead: FC(embed_dim→hidden_dim) + FC(hidden_dim→num_actions) — processes
                pre-extracted spatial features from ObservationEmbedding + GCNMHAStack

Tensor flow
-----------
  h_i   : (B, embed_dim)   — spatial feature for intersection i
  fc1   : (B, hidden_dim)  — ReLU(h_i · W_1 + b_1)
  fc2   : (B, num_actions) — q-values  (no activation; raw logits for MSE loss)

One QValueHead instance corresponds to ONE network in the RELight ensemble.
Each intersection has N independent QValueHead instances (the ensemble stack).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QValueHead(nn.Module):
    """
    Q-Value Prediction Head — fully-connected tail for the ABSTLight ensemble.

    Replaces the CNN + FC portion of the original RELight BaseDQN.  The input
    is the compact spatial feature vector h_{m_i}^L produced by the shared
    GCN-MHA front-end; this head projects it to a Q-value distribution over
    all traffic-signal actions for one specific intersection.

    Args:
        embed_dim   (int): Dimension of the incoming GCN-MHA output feature.
                           Must match GCNMHAStack.embed_dim.
        num_actions (int): Number of discrete traffic-signal phases / actions.
        hidden_dim  (int): Width of the intermediate fully-connected layer
                           (default 512, matching BaseDQN's fc1).
    """

    def __init__(self, embed_dim: int, num_actions: int, hidden_dim: int = 512):
        super(QValueHead, self).__init__()

        self.embed_dim = embed_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # FC layer 1: embed_dim → hidden_dim  (W_1, b_1)
        # Mirrors BaseDQN.fc1 → Linear(flatten_size, 512)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)

        # FC layer 2 (final projection): hidden_dim → num_actions  (W_p, b_p)
        # This is the   q = h^L · W_p + b_p   from the paper.
        self.fc2 = nn.Linear(hidden_dim, num_actions)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        """
        Kaiming initialisation for fc1 (feeds ReLU) and Xavier for fc2
        (feeds no activation — linear output layer).
        """
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict Q-values from a spatial feature vector.

        Args:
            h (torch.Tensor): GCN-MHA output feature for one intersection.
                Shape — Inference : (1, embed_dim)  or  (embed_dim,)
                         Training  : (B, embed_dim)

        Returns:
            q_values (torch.Tensor): Predicted Q-value for every action.
                Shape — same leading dims as h, last dim = num_actions.
                e.g. (B, num_actions) during training.

        Tensor dimension trace (training path):
            h        : (B, embed_dim)
            fc1+relu : (B, hidden_dim)   ← ReLU( h · W_1^T + b_1 )
            fc2      : (B, num_actions)  ← h_hidden · W_p^T + b_p
        """
        x = F.relu(self.fc1(h))      # (B, hidden_dim)
        q_values = self.fc2(x)       # (B, num_actions) — raw Q-value logits
        return q_values

    # ------------------------------------------------------------------
    # Action selection helper
    # ------------------------------------------------------------------

    def get_action(self, h: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection from this single Q-head.

        Intended for standalone debugging / inference.  During normal
        ABSTLight operation, action selection is handled by ABSTLightAgent
        which aggregates votes/averages across the full N-head ensemble.

        Args:
            h       (torch.Tensor): Feature vector, shape (1, embed_dim).
            epsilon (float)       : Exploration probability in [0, 1].

        Returns:
            action (int): Selected action index.
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_actions)

        with torch.no_grad():
            if h.dim() == 1:
                h = h.unsqueeze(0)          # (1, embed_dim)
            q_values = self.forward(h)
            return q_values.argmax(dim=1).item()

    # ------------------------------------------------------------------
    # Persistence helpers  (mirrors BaseDQN API)
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        """
        Save model weights and constructor metadata.

        Args:
            filepath (str): Destination .pth path.
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "embed_dim": self.embed_dim,
                "num_actions": self.num_actions,
                "hidden_dim": self.hidden_dim,
            },
            filepath,
        )

    def load(self, filepath: str):
        """
        Load model weights from a checkpoint saved by :meth:`save`.

        Args:
            filepath (str): Source .pth path.
        """
        checkpoint = torch.load(filepath, map_location="cpu")
        self.load_state_dict(checkpoint["state_dict"])

    def copy_weights_from(self, source: "QValueHead"):
        """
        In-place copy of weights from another QValueHead (mirrors BaseDQN API).

        Args:
            source (QValueHead): Source head to copy weights from.
        """
        self.load_state_dict(source.state_dict())

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}, "
            f"num_actions={self.num_actions}"
        )
