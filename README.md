# GPU-Accelerated Synthetic Dataset for Jenga Tower Physics

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Numba](https://img.shields.io/badge/Numba-JIT-orange)](https://numba.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> *"The model learned to see the past of a collapse by looking only at the rubble of the present."*

**Jenga Inverse Predictor v2 (JIP-2)** is a GPU-accelerated deep learning pipeline that treats structural anastylosis — the reassembly of collapsed stone monuments — as an **inverse physics problem**. Using Jenga towers as a physically tractable proxy for fallen architectural blocks, the notebook generates a fully synthetic dataset of collapse episodes, trains a multi-task ResNet-18, and reconstructs the original tower configuration from a single image of the ruins.

📄 **Paper:** [GPU-Accelerated Inverse Structural Anastylosis from Block Collapse Dynamics](https://github.com/LuisAlbertoMunozUbando/GPU-Accelerated-Inverse-Structural-Anastylosis-from-Block-Collapse-Dynamics)  
🏛️ **Institution:** School of Engineering and Sciences, Tecnológico de Monterrey, Mérida, Yucatán  
👤 **Author:** L. A. Muñoz Ubando · NVIDIA Ambassador

---

## How it works

```
Collapsed image  →  JIP-2 (ResNet-18 + friction injection)
                         ↓  predicts
                 blocks removed · positions · COM tilt · torque risk
                         ↓  retrieves
                 most similar simulated episode
                         ↓  renders
                 MP4: forward collapse → pause + overlay → reverse reconstruction
```

The key idea: fallen stone blocks, like fallen Jenga blocks, are governed by rigid-body mechanics. Their current positions **encode their dynamic history**. The model learns to decode that history from a photograph.

---

## Notebook structure

| Cell | Section | What it does |
|------|---------|--------------|
| 0 | **Setup & constants** | Writes `jenga_worker.py`, detects GPU/Numba, sets `DEMO_MODE`, defines Ziglar force thresholds for all 3 friction levels |
| 1 | **Import physics** | Imports all classes and functions from `jenga_worker`; runs a 3-layer smoke test |
| 2 | **Rendering** | 3-D matplotlib renderer using `Poly3DCollection`; produces per-frame PNG arrays |
| 3 | **Run 150 × 3 experiments** | Launches `ProcessPoolExecutor` with `spawn` context; runs 450 episodes across µ ∈ {0.25, 0.40, 0.60} in parallel; saves JSON results |
| 4 | **Videos + final images** | Renders MP4 collapse videos with `imageio`; saves final-state PNG per episode for CNN training |
| 5 | **Dashboard** | 12-panel Matplotlib figure: collapse rates, round distributions, Ziglar thresholds, COM margin histograms, Ziglar risk pie chart, torque% by friction level |
| 6 | **CNN definition** | `JengaCNN`: ResNet-18 backbone (grayscale-adapted) + friction one-hot injection + 4 prediction heads |
| 7 | **Training** | AdamW + CosineAnnealing; mixed-precision AMP on GPU; sklearn Ridge fallback without PyTorch |
| 8 | **Final summary** | Prints experiment totals, collapse rates, Ziglar stats, and hardware report |
| 9 | **Inference + reconstruction** | Loads any PNG → runs model → generates inference panel + reverse reconstruction MP4 |

---

## Physics engine (`jenga_worker.py`)

The notebook auto-generates `jenga_worker.py` in Cell 0. It contains the full rigid-body simulator used by the parallel workers.

### Pipeline

```
build_tower()  →  removable_blocks()  →  run_episode()
      ↓                                        ↓
 RigidBody[]                          frame_snapshots + fall_frames
                                              ↓
                              composite_collapse_check()
                              ziglar_block_analysis()
```

### Collision detection

```
GPU AABB broadphase (CuPy)  →  OBB SAT narrowphase (15 axes)  →  Face clipping (Sutherland–Hodgman)
```

Workers always use the CPU-NumPy version of AABB to avoid `cudaErrorInitializationError` on CUDA context inheritance via `fork()`.

### Contact solver

Projected Gauss-Seidel (PGS) compiled with `@njit(cache=True)`:

- **Normal impulse** with restitution e = 0.10
- **Coulomb friction cone** projection: λ_t clipped to ±µ_s λ_n
- **Baumgarte position correction**: β = 0.35, ε_slop = 0.4 mm
- **14 iterations** per timestep, 3 substeps at Δt = 1/240 s

### Collapse criterion (all 3 required simultaneously)

| Condition | Threshold |
|-----------|-----------|
| Max displacement | > 0.5 × block length |
| Kinetic energy | > 5 × 10⁻⁵ J |
| Max rotation angle | > 30° |

After collapse, `simulate_until_rest()` runs up to 400 additional steps until KE < 10⁻⁷ J.

### Ziglar thresholds (µ = 0.40, m = 19.6 g)

| Move type | Axis | F_min (mN) | Torque risk |
|-----------|------|------------|-------------|
| `center_xaxis` | X | 230.7 | No |
| `side_yaxis` | Y | 230.7 | No |
| `side_xaxis` | X | 307.6 | **Yes** |

Reference: *Ziglar, J. (2006). Analysis of Mechanics in Jenga. Robotics Institute, CMU.*

---

## Neural network (`JengaCNN`)

```
Input: grayscale image (1 × IMG_SIZE × IMG_SIZE)
           ↓
   ResNet-18 backbone
   (conv1 adapted: 3-ch → 1-ch by averaging ImageNet weights)
   4 ResBlock groups: 64 → 128 → 256 → 512 channels
           ↓
   128-dim visual embedding
           ↓  ‖  µ one-hot (3-dim) → Linear → 16-dim
   144-dim fused embedding  (ReLU + BN + Dropout 0.3)
           ↓
   ┌─ Head 1: num_removed   MSE   weight 0.4
   ├─ Head 2: removed_locs  BCE   weight 0.8  ← N_LAYERS×3 logits
   ├─ Head 3: imbalance     MSE   weight 0.4
   └─ Head 4: torque_risk   MSE   weight 0.4  ← Ziglar side_xaxis count
```

**Training details:**
- Optimiser: AdamW, lr = 3×10⁻⁴, weight_decay = 10⁻⁴
- Scheduler: CosineAnnealing, T_max = 30 epochs, η_min = lr / 20
- Augmentation: random rotations {0°, 90°, 180°, 270°} + horizontal flip
- Mixed precision: `torch.cuda.amp.autocast` + `GradScaler`
- Split: 80% train / 20% validation

**Fallback (no PyTorch):** sklearn `Ridge` regression trained on flattened grayscale pixels + friction one-hot. Saved to `models/sklearn_proxy.pkl`.

---

## Generated dataset

After running all cells, the `data/` directory contains:

```
data/
├── experiments/          # JSON per episode (positions, quats, Ziglar annotations)
├── frames/               # Final-state PNG per episode (CNN input)
│   └── nominal_exp_0042_final.png
├── videos/               # MP4 collapse videos
│   └── nominal_exp_0042.mp4
└── dashboard/
    └── ziglar_friction_dashboard.png
```

**Dataset size:** 450 episodes (150 per friction level × 3 levels), each with:
- Per-round block positions and quaternions
- `ziglar_risk` annotation per round (`low` / `medium` / `high`)
- `ziglar_can_torque` flag (True for `side_xaxis` moves)
- Full-floor contact frames after collapse
- Final-state image for CNN training

---

## Repository structure

```
.
├── A_GPU-Accelerated_Synthetic_Dataset_for_Jenga_Tower_Physics.ipynb
├── jenga_worker.py          # Auto-generated by Cell 0 — DO NOT edit manually
├── models/
│   ├── jenga_friction_cnn.pt    # ResNet-18 weights (PyTorch)
│   └── sklearn_proxy.pkl        # Ridge fallback (no PyTorch)
├── data/                    # Generated at runtime (see above)
└── README.md
```

> **Important:** `jenga_worker.py` is written automatically by Cell 0. It **must exist on disk before** the `ProcessPoolExecutor` is created in Cell 3. Spawn workers re-import modules by filename and cannot access functions defined inside the Jupyter kernel.

---

## Quick start

### Requirements

```bash
pip install numpy scipy matplotlib imageio[ffmpeg] pillow
pip install torch torchvision          # GPU training (recommended)
pip install cupy-cuda12x               # GPU AABB broadphase
pip install numba                      # JIT-compiled PGS solver
pip install psutil scikit-learn        # optional utilities
```

### Run

1. **Set mode** — `DEMO_MODE = True` for a fast 6-layer run (< 5 min, CPU-only) or `False` for the full 18-layer version (GPU recommended).

2. **Run cells in order** — Cell 0 must complete before Cell 3 (it writes `jenga_worker.py`).

3. **Run inference** on any fallen-tower image:

```python
# Cell 9
TEST_IMAGE_PATH = "data/frames/nominal_exp_0012_final.png"
TEST_MU_LEVEL   = "nominal"   # "low" | "nominal" | "high"
# Outputs: inference panel + reverse reconstruction MP4 in data/videos/
```

### Interpreting predictions

| Output | Typical range | Meaning |
|--------|---------------|---------|
| `num_removed` | 3 – 8 | Blocks removed before collapse |
| `removed_locs` | 0 – 1 per position | Probability each tower position was emptied |
| `imbalance_score` | 0 – 30 mm | How off-centre the COM was before collapse |
| `torque_risk` | 0 – 5 | Number of dangerous X-axis (Ziglar Eq. 8) moves |

---

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `cudaErrorInitializationError` | CUDA context inherited via `fork()` | Already handled: workers use 100% CPU via `spawn` context |
| `AttributeError: Can't get attribute 'run_episode'` | `jenga_worker.py` not on disk when pool starts | Run Cell 0 first; do not skip cells |
| `NameError` inside f-strings | Comprehension variable in nested f-string | Extract to auxiliary variable before the f-string |
| `No model found` | Training cell not executed | Run Cell 7 first; Ridge fallback activates automatically without PyTorch |
| `FileNotFoundError` on PNG | Cell 4 not completed | Run Cell 4 to generate final-state images before training |

---

## Physical parameters

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Block length | L | 8.1 cm | South (2003) |
| Block width | W | 2.6 cm | South (2003) |
| Block height | H | 1.8 cm | South (2003) |
| Block mass | m | 19.6 g | South (2003) |
| Static friction (nominal) | µ_s | 0.40 | Forest Products Lab (2002) |
| Restitution | e | 0.10 | Estimated, hardwood |
| Gravity | g | 9.81 m/s² | Standard |
| Timestep | Δt | 1/240 s | Design |
| Substeps | n_sub | 3 | Design |
| PGS iterations | n_it | 14–16 | Design |

---

## Connection to archaeological anastylosis

The pipeline is designed to transfer from Jenga to real-world structural anastylosis at the **UNESCO Maya site of Uxmal, Yucatán**. Practical deployment would require:

1. Replacing fixed-proportion Jenga blocks with a 3-D block detector calibrated from photogrammetric scans of the site
2. Enriching training data with synthetic collapse simulations of specific Puuc-style Maya wall typologies
3. Incorporating expert archaeological constraints (block orientation rules, mortar patterns) as additional loss terms

---

## Citation

```bibtex
@article{munoz2026jip2,
  title   = {GPU-Accelerated Inverse Structural Anastylosis
             from Block Collapse Dynamics},
  author  = {Mu{\~n}oz Ubando, L.~A.},
  journal = {Journal of Computational Something},
  year    = {2026},
  note    = {Tecnol{\'o}gico de Monterrey, M{\'e}rida, Yucat{\'a}n, M{\'e}xico}
}
```

---

## References

- Ziglar, J. (2006). *Analysis of Mechanics in Jenga*. Robotics Institute, Carnegie Mellon University.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. IEEE CVPR.
- South, M. (2003). *A Real-Time Physics Simulator for Jenga*. MSc thesis, University of Sheffield.
- Forest Products Laboratory (2002). *Wood Handbook — Wood as an Engineering Material*. USDA Forest Service.
- Muñoz et al. (2004). *Cultura Digital para la Zona Arqueológica de Uxmal*. UADY / INAH.
- Athens Charter (1931) · Venice Charter (1964) — foundational principles of anastylosis.
