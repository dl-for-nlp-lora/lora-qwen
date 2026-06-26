"""E3a: Grassmann subspace similarity between r=8 and r=64 LoRA adapters.

Reproduces Fig. 3 from Hu et al. 2021 (§7.2) on Qwen3-1.7B.

For each of 4 layers and each target module (q_proj, v_proj):
  - Load lora_A from the r=8 and r=64 checkpoints
  - SVD each -> right singular vectors (directions in 2048-dim input space)
  - Compute φ(i,j) = ‖U8[:,:i].T @ U64[:,:j]‖²_F / min(i,j)
    for i in 1..8, j in 1..64  (full grid, matching paper Fig. 3 x-axis)

Outputs (per module):
  E3a_rank_vs_rank_{module}.json  -- full 8×64 grid per layer
  (plotting is handled by E3_grassman_plot.py)

Usage:
    python results/E3/E3_grassman_script.py
"""

import json
import torch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "e3a_output"
OUT_DIR.mkdir(exist_ok=True)

CKPT_DIR = Path(r"C:\Users\HP\Downloads\checkpoints")
R8_PATH  = CKPT_DIR / "E2_r8_qv"  / "adapter.pt"
R64_PATH = CKPT_DIR / "E2_r64_qv" / "adapter.pt"

# ── Config ────────────────────────────────────────────────────────────────────
LAYERS  = [0, 9, 18, 27]           # 4 representative layers (of 28 total)
MODULES = ["q_proj", "v_proj"]
R_LOW   = 8                        # rank of the smaller checkpoint — i axis
R_HIGH  = 64                       # rank of the larger checkpoint  — j axis


# ── Core functions ────────────────────────────────────────────────────────────
def adapter_key(layer: int, module: str, tensor: str) -> str:
    return f"model.layers.{layer}.self_attn.{module}.lora.{tensor}"


def right_singular_vectors(lora_A: torch.Tensor) -> torch.Tensor:
    """Return right singular vectors of lora_A as columns.

    lora_A: (r, in_features)  ->  Vt: (r, in_features)  ->  V: (in_features, r)
    Columns of V are the top singular directions in input space, ordered by
    singular value (most important first).
    """
    _, _, Vt = torch.linalg.svd(lora_A.float(), full_matrices=False)
    return Vt.T  # (in_features, r)


def grassmann_phi(U_i: torch.Tensor, U_j: torch.Tensor) -> float:
    """Normalised subspace similarity (Hu et al. eq. 1).

    φ(i,j) = ‖U_i.T @ U_j‖²_F / min(i, j)  ∈ [0, 1]

    U_i: (in_features, i)  orthonormal columns from r=8 checkpoint
    U_j: (in_features, j)  orthonormal columns from r=64 checkpoint
    """
    M      = U_i.T @ U_j                               # (i, j)
    sigmas = torch.linalg.svdvals(M)                   # cosines of principal angles
    return (sigmas.pow(2).sum() / min(U_i.shape[1], U_j.shape[1])).item()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading adapter checkpoints...")
    sd8  = torch.load(R8_PATH,  map_location="cpu", weights_only=True)
    sd64 = torch.load(R64_PATH, map_location="cpu", weights_only=True)
    print(f"  r=8  keys: {len(sd8)}")
    print(f"  r=64 keys: {len(sd64)}")

    for module in MODULES:
        print(f"\nProcessing {module}...")
        result = {
            "module": module,
            "layers": [f"L{l}" for l in LAYERS],
            "i_max": R_LOW,    # i axis: 1..R_LOW  (r=8 directions)
            "j_max": R_HIGH,   # j axis: 1..R_HIGH (r=64 directions)
            "per_layer": {},
        }

        for layer in LAYERS:
            A8  = sd8 [adapter_key(layer, module, "lora_A")]  # (8,  2048)
            A64 = sd64[adapter_key(layer, module, "lora_A")]  # (64, 2048)

            U8  = right_singular_vectors(A8)   # (2048, 8)
            U64 = right_singular_vectors(A64)  # (2048, 64)

            # Full grid: i in 1..8, j in 1..64
            # grid[i-1][j-1] = phi(i, j)
            grid = []
            for i in range(1, R_LOW + 1):
                row = []
                for j in range(1, R_HIGH + 1):
                    row.append(grassmann_phi(U8[:, :i], U64[:, :j]))
                grid.append(row)
                print(f"  L{layer:2d}  i={i}/{R_LOW}  phi(i=1,j=1)={grid[0][0]:.4f}")

            result["per_layer"][f"L{layer}"] = grid

        out_path = OUT_DIR / f"E3a_rank_vs_rank_{module}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Wrote {out_path}")

    print("\nDone. Run E3_grassman_plot.py to generate figures.")


if __name__ == "__main__":
    main()