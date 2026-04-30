"""
weight_converter.py  —  Network Surgery: 80-D → 82-D Embedding Expansion
=========================================================================

Expands the observation-embedding weight matrix from 80-dimensional input
to 82-dimensional input by appending two zero-initialised columns.

Mathematical guarantee
----------------------
At fine-tuning step 0, every input to the network carries a context tag of
[0, 0] (all scenarios initialise their tag to zero-weighted values initially).
Because the new columns are exactly 0:

    new_embedding( cat([o_80D, 0, 0]) )
        = old_embedding(o_80D) + w[:, 80]*0 + w[:, 81]*0
        = old_embedding(o_80D)

So the network output is bit-for-bit identical to the base 80-D model,
fully preserving all prior knowledge before the first gradient step.

File I/O
--------
Input  : <src_dir>/embedding_episode_<N>.pth
         state_dict keys:
           "embedding.weight"  →  Tensor[embed_dim, 80]
           "embedding.bias"    →  Tensor[embed_dim]

Output : <src_dir>/finetune_ready_embedding.pth
         state_dict keys:
           "embedding.weight"  →  Tensor[embed_dim, 82]   (cols 80–81 = 0)
           "embedding.bias"    →  Tensor[embed_dim]        (unchanged)

Usage
-----
    # Default (episode 1000 in model/episode_1000/)
    python weight_converter.py

    # Custom source
    python weight_converter.py --src model/episode_1000 --episode 1000

    # Dry-run: inspect shapes without writing
    python weight_converter.py --dry-run
"""

import argparse
import sys
from pathlib import Path

import torch

# ============================================================================
# Constants
# ============================================================================

OLD_OBS_DIM: int = 80
CONTEXT_DIM: int = 2
NEW_OBS_DIM: int = OLD_OBS_DIM + CONTEXT_DIM   # 82
OUTPUT_FNAME: str = "finetune_ready_embedding.pth"

DEFAULT_SRC_DIR: str = "model/episode_1000"
DEFAULT_EPISODE: int = 1000


# ============================================================================
# Core conversion logic
# ============================================================================

def convert(src_dir: str, episode: int, dry_run: bool = False) -> None:
    """
    Load the pre-trained 80-D embedding checkpoint, zero-pad its weight matrix
    to 82 columns, and save the result as finetune_ready_embedding.pth.

    Args:
        src_dir  : Directory that contains embedding_episode_<episode>.pth.
        episode  : Episode number suffix of the source checkpoint.
        dry_run  : If True, print diagnostics but do NOT write any file.
    """
    src_path = Path(src_dir) / f"embedding_episode_{episode}.pth"
    dst_path = Path(src_dir) / OUTPUT_FNAME

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    if not src_path.exists():
        print(f"[ERROR] Source checkpoint not found:\n        {src_path.resolve()}")
        print(
            "\n  Make sure you have trained the base model to "
            f"episode {episode} and that MODEL_SAVE_DIR is 'model'."
        )
        sys.exit(1)

    if not dry_run and dst_path.exists():
        print(f"[WARN]  Output file already exists:\n        {dst_path.resolve()}")
        print(
            "        Delete it manually if you want to regenerate."
            "\n        Aborting to prevent accidental overwrite."
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    print(f"Loading : {src_path.resolve()}")
    old_state: dict = torch.load(src_path, map_location="cpu")

    if not isinstance(old_state, dict):
        print(
            f"[ERROR] Expected a state_dict (dict), got {type(old_state).__name__}.\n"
            "        Ensure the file was saved via ObservationEmbedding.state_dict()."
        )
        sys.exit(1)

    required_keys = {"embedding.weight", "embedding.bias"}
    missing = required_keys - old_state.keys()
    if missing:
        print(
            f"[ERROR] Missing keys in checkpoint: {missing}\n"
            f"        Available keys: {list(old_state.keys())}"
        )
        sys.exit(1)

    old_weight: torch.Tensor = old_state["embedding.weight"]  # [embed_dim, 80]
    old_bias:   torch.Tensor = old_state["embedding.bias"]    # [embed_dim]

    # Validate shape
    if old_weight.ndim != 2 or old_weight.shape[1] != OLD_OBS_DIM:
        print(
            f"[ERROR] Unexpected weight shape {list(old_weight.shape)}.\n"
            f"        Expected [embed_dim, {OLD_OBS_DIM}]."
        )
        sys.exit(1)

    embed_dim: int = old_weight.shape[0]

    # ------------------------------------------------------------------
    # Surgery: construct new [embed_dim, 82] weight matrix
    # ------------------------------------------------------------------
    #   new_weight[:, :80]   = pre-trained values  (all knowledge preserved)
    #   new_weight[:, 80:82] = 0.0                 (rain_flag, accident_flag)
    # ------------------------------------------------------------------
    new_weight = torch.zeros(embed_dim, NEW_OBS_DIM, dtype=old_weight.dtype)
    new_weight[:, :OLD_OBS_DIM] = old_weight
    # columns [:, 80] and [:, 81] remain exactly 0.0 by torch.zeros()

    new_state = {
        "embedding.weight": new_weight,
        "embedding.bias":   old_bias,
    }

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  ABSTLight Embedding Surgery  —  Summary")
    print("=" * 60)
    print(f"  Source episode    : {episode}")
    print(f"  Embed dim         : {embed_dim}")
    print(f"  Old weight shape  : {list(old_weight.shape)}  ({OLD_OBS_DIM}-D input)")
    print(f"  New weight shape  : {list(new_weight.shape)}  ({NEW_OBS_DIM}-D input)")
    print(f"  Dtype             : {old_weight.dtype}")
    print(f"  New columns [80, 81] (rain_flag, accident_flag): all zeros")
    print()

    # Sanity-check: with a zero context tag the outputs must be identical
    _verify_output_identity(old_state, new_state, embed_dim, dry_run)

    if dry_run:
        print("[DRY-RUN] No file written.")
        return

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    torch.save(new_state, dst_path)
    print(f"[OK]  Saved : {dst_path.resolve()}")
    print()
    print("  Next step: run train_finetune.py")
    print("  The gateway check will locate this file automatically.")


def _verify_output_identity(
    old_state: dict,
    new_state: dict,
    embed_dim: int,
    dry_run: bool,
) -> None:
    """
    Quick algebraic correctness check:
    Verify that new_embedding([obs_80d, 0, 0]) == old_embedding(obs_80d)
    for a random probe vector.
    """
    try:
        # Import lazily so the script works even if model_components is unavailable
        from model_components.observation_embedding import ObservationEmbedding

        old_net = ObservationEmbedding(obs_dim=OLD_OBS_DIM, embed_dim=embed_dim)
        old_net.load_state_dict(old_state)
        old_net.eval()

        new_net = ObservationEmbedding(obs_dim=NEW_OBS_DIM, embed_dim=embed_dim)
        new_net.load_state_dict(new_state)
        new_net.eval()

        probe = torch.randn(4, OLD_OBS_DIM)                    # (4, 80) random batch
        probe_padded = torch.cat(
            [probe, torch.zeros(4, CONTEXT_DIM)], dim=1
        )                                                       # (4, 82)  tag=[0,0]

        with torch.no_grad():
            out_old = old_net(probe)
            out_new = new_net(probe_padded)

        max_diff = float((out_old - out_new).abs().max().item())
        if max_diff < 1e-5:
            print(f"  Identity check    : PASSED  (max |diff|={max_diff:.2e})")
        else:
            print(
                f"  Identity check    : FAILED  (max |diff|={max_diff:.2e})\n"
                "  [WARN] Output diverges — check weight surgery logic."
            )
    except ImportError:
        print("  Identity check    : SKIPPED (model_components not importable here)")
    print()


# ============================================================================
# CLI entry point
# ============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand ABSTLight ObservationEmbedding from "
            f"{OLD_OBS_DIM}-D to {NEW_OBS_DIM}-D for 3-in-1 fine-tuning."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src",
        default=DEFAULT_SRC_DIR,
        help=f"Directory containing embedding_episode_<episode>.pth.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=DEFAULT_EPISODE,
        help="Episode number suffix of the source checkpoint to convert.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diagnostics and run identity check without writing any file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert(src_dir=args.src, episode=args.episode, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
