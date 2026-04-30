"""
GCN-MHA Layer  —  ABSTLight Layer 2
=====================================

Implements the Graph Convolutional Network + Multi-Head Attention (GCN-MHA)
layer from the ABSTLight paper. This layer is the spatial core of the model:
it lets each intersection aggregate weighted information from its road-network
neighbours to build a spatially-aware representation.

Mathematical specification
--------------------------
For a single attention head h:

    head_h = Attention(H·W_h^Q,  H·W_h^K,  H·W_h^V)

where the scaled-dot-product attention is:

    Attention(Q, K, V) = softmax( Q·K^T / sqrt(d_k) ) · V

The graph topology is enforced by masking out non-neighbour positions in the
attention score matrix before the softmax so that each intersection only
attends to its direct road-network peers (and itself).

The multi-head output:

    MultiHead(Q, K, V) = Concat(head_1, ..., head_H) · W^O

This module uses PyTorch's built-in nn.MultiheadAttention which fuses all
Q/K/V projections and the scaled dot-product computation into a single,
numerically stable kernel.

Module summary
--------------
  GCNMHALayer   — one graph-attention layer (with residual + LayerNorm)
  GCNMHAStack   — L stacked GCNMHALayer instances

Tensor flow (training path)
---------------------------
    H^0        = ObservationEmbedding output   (B, N, D)
    H^1        = GCNMHALayer_1(H^0, adj)       (B, N, D)
    ...
    H^L        = GCNMHALayer_L(H^{L-1}, adj)   (B, N, D)  ← final output

    H^L[:, i, :]  encodes intersection i's collaborative representation after
    aggregating information from its L-hop road-network neighbourhood.

Key conventions
---------------
  B  : replay-buffer batch size
  N  : number of intersections (graph nodes)
  D  : embed_dim (must be divisible by num_heads)
  H_ : number of attention heads
  dk : D / H_  (dimension per head)

Reference
---------
ABSTLight paper, GCN-MHA section:
    "A multi-layer graph attention network is used to determine the importance
     of neighbouring intersections. MHA processes different sub-spaces
     simultaneously to capture collaborative information."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Single GCN-MHA layer
# ---------------------------------------------------------------------------

class GCNMHALayer(nn.Module):
    """
    Single GCN-MHA Layer — one graph-attention step over the intersection graph.

    Architecture per forward call
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Input  H         : (B, N, D)
        ↓  Adjacency mask built from adj  : (N, N)  { 0 / -inf }
        ↓  nn.MultiheadAttention (Q=K=V=H): (B, N, D)
        ↓  Residual connection + LayerNorm: (B, N, D)
        Output H_out     : (B, N, D)

    The residual connection (Transformer-style skip) and LayerNorm stabilise
    gradients when several GCNMHALayers are stacked.

    Args:
        embed_dim      (int)  : Feature dimension D.  Must be divisible by num_heads.
        num_heads      (int)  : Number of parallel attention heads H_.
        dropout        (float): Dropout probability applied inside MHA.
        add_self_loops (bool) : If True, the diagonal of adj is forced to 1 so
                                every intersection always attends to itself.
                                Prevents all-−inf attention rows for isolated nodes.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        add_self_loops: bool = True,
    ):
        super(GCNMHALayer, self).__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.add_self_loops = add_self_loops

        # nn.MultiheadAttention internally holds:
        #   in_proj_weight  (3·D, D)  — merged W^Q, W^K, W^V
        #   in_proj_bias    (3·D,)
        #   out_proj.weight (D, D)    — W^O
        #   out_proj.bias   (D,)
        # batch_first=True means input/output are (B, N, D) rather than (N, B, D).
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Post-attention residual normalisation (Transformer convention).
        self.layer_norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    # Helper: adjacency → additive attention mask
    # ------------------------------------------------------------------

    @staticmethod
    def _adj_to_attn_mask(
        adj: torch.Tensor,
        add_self_loops: bool,
    ) -> torch.Tensor:
        """
        Convert a binary adjacency matrix to an additive attention mask.

        nn.MultiheadAttention adds this mask to the raw attention logits
        *before* softmax.  The convention is:
            0.0    → "attend to this position" (logit carried through)
            -inf   → "do not attend" (logit → −∞, softmax → 0)

        Args:
            adj           (torch.Tensor): (N, N) binary adjacency.
                                          adj[i, j] = 1  iff i and j share a road.
            add_self_loops (bool)       : Force diagonal to 1 before masking.

        Returns:
            mask (torch.Tensor): (N, N) float32 with values in {0.0, -inf}.

        Dimension trace:
            adj    : (N, N)  int / bool / float01
            a      : (N, N)  float32, optionally with diagonal = 1
            mask   : (N, N)  float32  {0 where connected, -inf where not}
        """
        N = adj.size(0)
        a = adj.float().clone()

        if add_self_loops:
            # Ensure every node attends to itself regardless of the input adj.
            eye = torch.eye(N, device=adj.device, dtype=a.dtype)
            a = (a + eye).clamp(max=1.0)  # keep binary

        # mask[i, j] = -inf  when a[i, j] == 0 (non-neighbour / non-self)
        # mask[i, j] =  0.0  when a[i, j] == 1 (neighbour or self)
        mask = torch.zeros(N, N, device=adj.device, dtype=torch.float32)
        mask[a == 0.0] = float("-inf")
        return mask  # (N, N)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        H: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        One graph-attention step.

        Args:
            H   (torch.Tensor): Hidden states from the previous layer.
                Shape — Inference : (N, D)
                         Training  : (B, N, D)
            adj (torch.Tensor): Binary adjacency matrix, shape (N, N).
                adj[i, j] = 1 iff intersection i has a direct road connection
                to intersection j.  Can be directed or undirected.

        Returns:
            H_out (torch.Tensor): Updated hidden states, same shape as H.

        Tensor dimension trace (training path):
            H            : (B, N, D)
            attn_mask    : (N, N)      ← built from adj
            attn_output  : (B, N, D)  ← nn.MHA output, (Q=K=V=H, masked)
            residual     : (B, N, D)  ← H + attn_output
            H_out        : (B, N, D)  ← LayerNorm(residual)
        """
        # ---- Handle unbatched inference input (N, D) ------------------
        # Add a batch dimension so nn.MultiheadAttention always sees (B, N, D).
        squeeze_out = H.dim() == 2
        if squeeze_out:
            H = H.unsqueeze(0)          # (1, N, D)

        B, N, D = H.shape

        # ---- Build adjacency-based attention mask --------------------
        # Shape: (N, N) — reused for all B samples and all H_ heads.
        attn_mask = self._adj_to_attn_mask(adj, self.add_self_loops)
        # attn_mask dtype must match H dtype for MHA (promote if needed).
        attn_mask = attn_mask.to(dtype=H.dtype, device=H.device)

        # ---- Multi-Head Self-Attention --------------------------------
        # Q = K = V = H: each node queries its own representation against
        # its neighbours' representations to produce an attention-weighted
        # aggregation (the "mean-field" step from the paper).
        #
        # attn_output[b, i, :] = Σ_j  α(i,j) · (h_j · W^V)
        # where α(i,j) ∝ exp(score(i,j)) · adj[i,j]
        #
        # nn.MultiheadAttention signature (batch_first=True):
        #   forward(query, key, value, attn_mask) → (output, attn_weights)
        #
        # output shape: (B, N, D)
        attn_output, _ = self.mha(
            query=H,
            key=H,
            value=H,
            attn_mask=attn_mask,
            need_weights=False,
        )

        # ---- Residual + LayerNorm ------------------------------------
        # Stabilises deep stacks; mirrors the Transformer encoder block.
        # H_out shape: (B, N, D)
        H_out = self.layer_norm(H + attn_output)

        # ---- Restore original shape if needed ------------------------
        if squeeze_out:
            H_out = H_out.squeeze(0)    # (N, D)

        return H_out

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"add_self_loops={self.add_self_loops}"
        )


# ---------------------------------------------------------------------------
# Stacked GCN-MHA layers
# ---------------------------------------------------------------------------

class GCNMHAStack(nn.Module):
    """
    Stacked GCN-MHA Layers — the complete spatial feature extractor (Layer 2).

    Runs the embedded hidden states through `num_layers` GCNMHALayer instances
    sequentially.  Each layer refines the intersection representations by
    attending over road-network neighbours with a progressively wider
    receptive field (L layers ≈ L-hop neighbourhood).

    Tensor flow (training path)
    ---------------------------
        H^0  = ObservationEmbedding output        (B, N, embed_dim)
        H^1  = GCNMHALayer_1(H^0, adj)            (B, N, embed_dim)
        H^2  = GCNMHALayer_2(H^1, adj)            (B, N, embed_dim)
        ...
        H^L  = GCNMHALayer_L(H^{L-1}, adj)        (B, N, embed_dim)

    H^L[:, i, :]  is passed to QValueHead[i] as the spatially-informed
    representation of intersection i after aggregating information from its
    L-hop neighbourhood.

    Args:
        embed_dim      (int)  : Feature dimension — must match
                                ObservationEmbedding.embed_dim.
        num_heads      (int)  : Attention heads per layer.
        num_layers     (int)  : Depth L (number of stacked GCNMHALayer).
        dropout        (float): Per-layer dropout probability (forwarded).
        add_self_loops (bool) : Forwarded to each GCNMHALayer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        add_self_loops: bool = True,
    ):
        super(GCNMHAStack, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                GCNMHALayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        H: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run all L GCNMHALayers sequentially.

        Args:
            H   (torch.Tensor): Embedded hidden states from ObservationEmbedding.
                Shape — Inference : (N, embed_dim)
                         Training  : (B, N, embed_dim)
            adj (torch.Tensor): Binary adjacency matrix, shape (N, N).
                Constant across every call (same road network topology).

        Returns:
            H_L (torch.Tensor): Final spatial representations, same shape as H.
                H_L[..., i, :] is the collaborative embedding of intersection i
                incorporating information from its L-hop neighbourhood.

        Tensor dimension trace (training path):
            H_0        : (B, N, embed_dim)   ← from ObservationEmbedding
            H_1        : (B, N, embed_dim)   ← after layer 1
            ...
            H_num_layers : (B, N, embed_dim) ← returned
        """
        for layer in self.layers:
            H = layer(H, adj)
        return H

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"num_layers={self.num_layers}"
        )
