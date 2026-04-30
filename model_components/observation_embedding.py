"""
Observation Embedding Layer  —  ABSTLight Layer 1
==================================================

Implements the first processing stage of the ABSTLight architecture:

    h_i = σ(o_i · W_e + b_e)

where:
  o_i   — raw traffic observation vector for intersection i, shape (obs_dim,)
  W_e   — learnable weight matrix, shape (obs_dim, embed_dim)   [stored transposed by nn.Linear]
  b_e   — learnable bias vector, shape (embed_dim,)
  σ     — ReLU activation
  h_i   — embedded hidden state for intersection i, shape (embed_dim,)

The module is dimension-agnostic: the same nn.Linear call broadcasts cleanly
over any number of leading dimensions, so no reshaping is needed between the
single-step inference shape and the batched training shape.

Typical tensor flow
-------------------
  Inference  : (N_agents, obs_dim)      →  (N_agents, embed_dim)
  Training   : (B, N_agents, obs_dim)   →  (B, N_agents, embed_dim)

Reference
---------
ABSTLight paper, Section on Observation Embedding:
    "The observation o_i^t of agent i at time t is input to the embedding
     layer to obtain the intersection's hidden state h_i."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationEmbedding(nn.Module):
    """
    Observation Embedding Layer — maps raw traffic observations to hidden states.

    A single linear transformation followed by ReLU (σ) projects the raw
    per-intersection observation vector o_i into a compact hidden state h_i
    that is subsequently consumed by the GCN-MHA layer.

    Because nn.Linear broadcasts over leading batch / agent dimensions the
    same object handles both inference-time (N, D) tensors and training-time
    (B, N, D) tensors without any reshaping.

    Args:
        obs_dim   (int): Dimension of the raw observation vector per
                         intersection. Includes traffic flow counts per lane/
                         approach plus signal phase indicators.
                         Example breakdown for a 4-approach × 2-lane intersection:
                           - 8  lane vehicle queue counts
                           - 8  lane mean waiting times
                           - 4  current signal phase (one-hot)
                         → obs_dim = 20

        embed_dim (int): Dimension of the output hidden state h_i.
                         Must match the `embed_dim` used in GCNMHAStack and
                         QValueHead.  Typically 64 or 128.
    """

    def __init__(self, obs_dim: int, embed_dim: int):
        super(ObservationEmbedding, self).__init__()

        self.obs_dim = obs_dim
        self.embed_dim = embed_dim

        # Single linear layer  (W_e, b_e from the paper).
        # nn.Linear stores weights as (embed_dim, obs_dim) and computes:
        #   output = input @ W_e^T + b_e
        # which is algebraically identical to the paper's  o_i · W_e + b_e.
        self.embedding = nn.Linear(obs_dim, embed_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        """
        Kaiming (He) initialisation, appropriate for ReLU activations.

        Sets weight with fan-out mode to keep activation variance stable
        through deep networks; biases start at zero.
        """
        nn.init.kaiming_normal_(
            self.embedding.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.embedding.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute h_i = ReLU( o_i · W_e + b_e ) for every intersection.

        Args:
            obs (torch.Tensor): Raw traffic observations.
                Shape — Inference : (N_agents, obs_dim)
                         Training  : (B, N_agents, obs_dim)
                where B is the replay-buffer batch size and N_agents is the
                number of intersections in the road network.

        Returns:
            h (torch.Tensor): Embedded hidden states.
                Shape — Inference : (N_agents, embed_dim)
                         Training  : (B, N_agents, embed_dim)
                The leading dimensions are preserved identically.

        Tensor dimension trace (training path):
            obs   : (B, N, obs_dim)
            linear: (B, N, embed_dim)   ← W_e, b_e applied to last axis only
            relu  : (B, N, embed_dim)   ← element-wise, shape unchanged
        """
        # nn.Linear broadcasts over all leading dimensions, so both the
        # (N, obs_dim) and (B, N, obs_dim) paths go through the same call.
        h = F.relu(self.embedding(obs))
        return h

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return f"obs_dim={self.obs_dim}, embed_dim={self.embed_dim}"
