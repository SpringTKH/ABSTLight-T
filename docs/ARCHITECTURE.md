# Algorithm and Network Topology

## Layer 1 — Observation Embedding

**Module:** `ObservationEmbedding` (`model_components/observation_embedding.py`)

The raw observation tensor is first projected into a compact latent space via a single affine transformation followed by a ReLU non-linearity:

$$h_i = \sigma\!\left(\mathbf{o}_i \cdot W_e + \mathbf{b}_e\right), \quad W_e \in \mathbb{R}^{D_\text{obs} \times D_\text{embed}}$$

```python
self.embedding = nn.Linear(obs_dim, embed_dim)   # obs_dim=80 (or 82), embed_dim=64
```

**Weight initialisation:** Kaiming (He) normal with `mode="fan_out"` for the weight matrix; zero bias. This preserves activation variance through ReLU activations in deep networks.

**Tensor dimension trace:**

| Context   | Input shape  | Output shape |
| --------- | ------------ | ------------ |
| Inference | $(N, 80)$    | $(N, 64)$    |
| Training  | $(B, N, 80)$ | $(B, N, 64)$ |

The layer broadcasts over the leading batch dimension without any explicit reshaping.

## Layer 2 — Graph Convolutional Multi-Head Attention (GCN-MHA)

**Module:** `GCNMHAStack` / `GCNMHALayer` (`model_components/gcn_mha.py`)

The embedded representations are passed through a stack of $L = 2$ graph-masked multi-head attention layers. Each layer implements the spatial aggregation operation:

$$H^{(l)} = \text{LayerNorm}\!\left(H^{(l-1)} + \text{MHA}\!\left(Q=H^{(l-1)},\, K=H^{(l-1)},\, V=H^{(l-1)};\; \text{mask}=\mathcal{M}\right)\right)$$

**Graph topology enforcement.** The road-network adjacency matrix $A \in \{0,1\}^{N \times N}$ (derived at simulation initialisation from `traci.trafficlight.getControlledLanes`) is converted to an additive attention mask $\mathcal{M}$ that suppresses attention between non-adjacent intersections:

```python
@staticmethod
def _adj_to_attn_mask(adj, add_self_loops):
    a = adj.float().clone()
    if add_self_loops:
        a = (a + torch.eye(N, device=adj.device)).clamp(max=1.0)
    mask = torch.zeros(N, N, device=adj.device, dtype=torch.float32)
    mask[a == 0.0] = float("-inf")   # non-neighbours → −∞ before softmax
    return mask                       # shape: (N, N)
```

The mask is passed directly into PyTorch's `nn.MultiheadAttention`:

```python
attn_output, _ = self.mha(
    query=H, key=H, value=H,
    attn_mask=attn_mask,
    need_weights=False
)
H_out = self.layer_norm(H + attn_output)
```

**Architectural parameters:**

| Parameter                            | Value | Config Key            |
| ------------------------------------ | ----- | --------------------- |
| Embedding dimension $D$              | 64    | `EMBED_DIM = 64`      |
| Number of attention heads $H'$       | 4     | `NUM_HEADS = 4`       |
| Dimension per head $d_k = D / H'$    | 16    | — (derived)           |
| Number of stacked GCN-MHA layers $L$ | 2     | `NUM_GCN_LAYERS = 2`  |
| Attention dropout                    | 0.1   | `GCN_DROPOUT = 0.1`   |
| Self-loop augmentation               | True  | `add_self_loops=True` |

**Internal weight matrices per layer** (`nn.MultiheadAttention`):

- Fused in-projection: $W^{Q,K,V} \in \mathbb{R}^{3D \times D} = \mathbb{R}^{192 \times 64}$
- Output projection: $W^O \in \mathbb{R}^{D \times D} = \mathbb{R}^{64 \times 64}$
- Post-attention LayerNorm: $\gamma, \beta \in \mathbb{R}^{64}$

**Tensor dimension trace:**

```
H^0 = ObservationEmbedding output  →  (B, N, 64)
H^1 = GCNMHALayer_1(H^0, adj)      →  (B, N, 64)
H^2 = GCNMHALayer_2(H^1, adj)      →  (B, N, 64)   ← final spatial features
```

The output $H^{(L)}[:, i, :]$ encodes the spatially-collaborative representation of intersection $i$ after aggregating information from all 2-hop road-network neighbours.

## Layer 3 — Ensemble Q-Networks (RELight)

**Module:** `QValueHead` (`model_components/q_value_head.py`) + `ABSTLightEnsembleUpdater` (`agent/abst_light_agent.py`)

Each intersection $i$ maintains an independent ensemble of $N = 10$ Q-networks (`q_heads[agent_idx][network_idx]`). Each `QValueHead` is a two-layer fully-connected network:

$$q(h_i) = \underbrace{FC_{512}(h_i)}_{\text{fc1: }64 \to 512} \xrightarrow{\text{ReLU}} \underbrace{FC_{4}}_{\text{fc2: }512 \to 4}$$

```python
self.fc1 = nn.Linear(embed_dim,  hidden_dim)   # 64  → 512
self.fc2 = nn.Linear(hidden_dim, num_actions)  # 512 → 4
```

Output: Q-values over $|\mathcal{A}| = 4$ discrete traffic-signal phases (no final activation; raw logits for MSE/Huber loss).

**Weight initialisation:** Kaiming normal for `fc1` (ReLU-fed); Xavier uniform for `fc2` (linear output).

**Ensemble summary:**

| Quantity                     | Value             | Config Key                |
| ---------------------------- | ----------------- | ------------------------- | --- | ----------------- |
| Q-heads per intersection $N$ | 10                | `N_NETWORKS = 10`         |
| Target subset size $M$       | 4                 | `M_SUBSET_SIZE = 4`       |
| Number of actions $          | \mathcal{A}       | $                         | 4   | `NUM_ACTIONS = 4` |
| Discount factor $\gamma$     | 0.95              | `GAMMA`                   |
| Optimizer                    | Adam (all params) | —                         |
| Hidden layer width           | 512               | `Q_HEAD_HIDDEN_DIM = 512` |

**Optimiser structure.** The shared front-end (ObservationEmbedding + GCNMHAStack) and each `QValueHead` have independent Adam optimisers. This allows the shared trunk to accumulate gradients from all $(N_\text{agents} \times N_\text{heads})$ loss terms before a single backward pass:

```python
self.frontend_optimizer = optim.Adam(
    list(embedding.parameters()) + list(gcn_mha.parameters()),
    lr=learning_rate
)
self.head_optimizers[i][k] = optim.Adam(q_heads[i][k].parameters(), lr=learning_rate)
```

## Conservative Bellman Target via Min-Pooling ($M$-Subset Sampling)

During training, the RELight algorithm avoids over-optimistic Q-value estimation by computing the Bellman target using the **minimum Q-value across an $M$-subset** of ensemble heads, sampled uniformly without replacement:

```python
# agent/abst_light_agent.py  ·  _compute_target_q()
subset_indices = random.sample(range(self.N), self.M)   # M=4 of N=10 heads

max_q_per_head = [
    self.q_heads[agent_idx][k](h_next).max(dim=1).values   # (B,) per head
    for k in subset_indices
]

# Stack → (M, B), then element-wise minimum across heads → (B,)
q_min = torch.stack(max_q_per_head, dim=0).min(dim=0).values   # (B,)

scaled_rewards = rewards / self.reward_scale_divisor
target_q = scaled_rewards + self.gamma * q_min * (1.0 - dones)
```

This produces a **conservative target** that penalises optimistic over-estimation, improving stability under the non-stationary traffic demand distribution.

An **adaptive error threshold** gates subset resampling: if the initial subset yields an MSE error exceeding the threshold, a fresh subset is drawn (up to `max_resampling_attempts = 5` times) to prevent degenerate gradient updates.

## Loss Function and Gradient Descent

### Loss Function — Huber Loss (`nn.SmoothL1Loss`)

The training objective uses PyTorch's **`nn.SmoothL1Loss`** (Huber Loss), chosen over plain MSE for robustness to occasional large TD errors that occur during early exploration:

$$\mathcal{L}(q_\text{pred}, q_\text{target}) = \begin{cases} \tfrac{1}{2}(q_\text{pred} - q_\text{target})^2 & \text{if } |q_\text{pred} - q_\text{target}| < 1 \\ |q_\text{pred} - q_\text{target}| - \tfrac{1}{2} & \text{otherwise} \end{cases}$$

```python
# agent/abst_light_agent.py  ·  ABSTLightEnsembleUpdater.__init__()
# Huber loss is more robust than MSE for occasional large TD errors.
self.criterion = nn.SmoothL1Loss()
```

The loss is computed per `(agent i, Q-head k)` pair against the conservative RELight target, then all per-head scalar losses are **summed** into a single total loss before the backward pass:

```python
# Step 3: per (i, k) pair
loss = self.criterion(q_pred, q_target_final)   # scalar Huber loss
losses.append(loss)

# Step 4: single joint backward pass
total_loss = torch.stack(losses).sum()
total_loss.backward()
```

This single `backward()` call allows the **shared front-end** (ObservationEmbedding + GCNMHAStack) to accumulate gradients from all $(N_\text{agents} \times N_\text{heads})$ loss terms simultaneously, while each `QValueHead`'s gradient flows only through its own `fc1`/`fc2` parameters.

### Optimiser — Adam

All optimisers are `torch.optim.Adam` with a uniform learning rate. No explicit `betas`, `eps`, or `weight_decay` arguments are passed, so PyTorch default values apply throughout:

```python
# Shared front-end (embedding + gcn_mha)
self.frontend_optimizer = optim.Adam(frontend_params, lr=learning_rate)

# Per-agent, per-Q-head heads
self.head_optimizers[i][k] = optim.Adam(q_heads[i][k].parameters(), lr=learning_rate)
```

| Hyperparameter                             | Value          | Source                                |
| ------------------------------------------ | -------------- | ------------------------------------- |
| Learning rate $\alpha$                     | `1e-4`         | `config.py: LEARNING_RATE = 0.0001`   |
| Momentum coefficients $(\beta_1, \beta_2)$ | `(0.9, 0.999)` | PyTorch Adam default                  |
| Numerical stability $\varepsilon$          | `1e-8`         | PyTorch Adam default                  |
| Weight decay (L2 regularisation)           | `0` (none)     | PyTorch Adam default — not overridden |
