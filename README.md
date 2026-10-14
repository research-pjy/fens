# FENS: Revisiting Ensembling in One-Shot Federated Learning (NeurIPS 2024)

A clean, modular, from-scratch PyTorch implementation of **FENS** (*Revisiting Ensembling in One-Shot Federated Learning*, NeurIPS 2024) structured in standard Python `src/` layout (`src/fens/`) and optimized for execution on the **`rama` NVIDIA L40S (44 GiB VRAM)** server inside the `qwen-vl` Conda environment.

---

## Repository Structure (`src/` Layout)

```text
fens-reproduction/
├── pyproject.toml                         # Standard PEP-621 package config (src/ layout)
├── README.md                              # Architecture & execution instructions
├── src/
│   └── fens/
│       ├── __init__.py
│       ├── config.py                      # Paper hyperparameters (Tables 6 & 8) + isolated L40S HardwareConfig
│       ├── data/
│       │   ├── partition.py               # Dirichlet Dir_M(alpha) + Custom 2-Client Generalist-vs-Specialist split
│       │   └── datasets.py                # CIFAR-10, SVHN, and CIFAR-100 (auxiliary KD) DataLoaders
│       ├── models/
│       │   ├── resnet8.py                 # Standard ResNet-8 (18.85 MiB) & Downsized ResNet-8-DS (0.88 MiB, Table 4)
│       │   └── quantization.py            # Server-side FP32 -> INT8 post-training quantization (Algorithm 1, Line 6)
│       ├── aggregators/
│       │   └── models.py                  # All 7 paper aggregators: Uniform, Weighted, Linear, Per-Class, MLP, Voting, MoE
│       ├── fl/
│       │   ├── phase1_local.py            # Algorithm 2 (Lines 2-5): Phase 1 one-shot local training & checkpointing
│       │   ├── phase2_aggregator.py       # Algorithm 1 & 2: Phase 2 FedAdam aggregator training with L40S logit caching
│       │   └── baselines.py               # One-Shot (FedAvg-1R, FedEns, FedKD), Iterative FL (FedAdam), and Server KD
│       └── utils/
│           ├── hardware.py                # L40S (`rama`) GPU/VRAM/BF16 verifier & deterministic seeding
│           ├── comm_cost.py               # Exact byte/MiB/GiB communication cost calculator & OFL normalization
│           └── plotting.py                # Publication-quality plots matching Figures 2, 4, 8, 9 + 2-Client Skew diagnostics
├── scripts/
│   ├── 00_verify_and_smoke_test.py        # Pre-training L40S hardware & end-to-end pipeline smoke test (<15 sec)
│   ├── 01_train_phase1_clients.py         # Step 1: Pre-train & save Phase 1 local client checkpoints
│   ├── 02_run_phase2_fens_and_baselines.py# Step 2: Live demo evaluation of FENS Phase 2, baselines & Figure 2/8 plots
│   └── 03_run_two_client_skew_demo.py     # Custom Experiment: Client 0 (Balanced 10-Class) vs. Client 1 (1-2 Class Specialist)
└── tests/
    └── test_fens_core.py                  # Unit test suite (`pytest`)
```

---

## Development & Execution Workflow (`Local Ubuntu -> GitHub -> rama L40S`)

### 1. Local Development Machine
```bash
# Run unit tests locally
PYTHONPATH=src pytest -v

# Initialize Git and push to your GitHub repository
git init
git add .
git commit -m "Implement FENS from scratch in src/ layout with L40S support"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

### 2. Remote Execution on `rama` (NVIDIA L40S)
Log into `rama`, clone/pull your repository, and activate the existing `qwen-vl` Conda environment (no environment modifications required):

```bash
conda activate qwen-vl
git pull origin main

# Optional editable install without modifying existing packages:
pip install -e . --no-deps
```

#### Step 0: Run the Mandatory L40S Smoke Test First
Verifies that `rama`'s L40S GPU and BF16 support are detected and exercises data partitioning, ResNet-8 forward/backward pass, finite loss/gradients, FP32-to-INT8 quantization, Phase 2 `FedAdam` updates, and checkpoint saving in under 15 seconds:
```bash
PYTHONPATH=src python scripts/00_verify_and_smoke_test.py
```

#### Step 1: Pre-Train & Checkpoint Phase 1 Client Models (Run Before Live Demo)
Trains the $M=20$ client models on their 90% local splits ($D_{i1}$) and saves checkpoints to `checkpoints/phase1/` plus the Figure 9 Dirichlet bubble chart:
```bash
# Full paper settings (CIFAR-10, 500 local epochs across alpha in {0.01, 0.05, 0.1}):
PYTHONPATH=src python scripts/01_train_phase1_clients.py --dataset cifar10 --alphas 0.01 0.05 0.1

# Or SVHN (paper default is 50 local epochs, which finishes very quickly on L40S):
PYTHONPATH=src python scripts/01_train_phase1_clients.py --dataset svhn --alphas 0.01 0.05 0.1
```

#### Step 2: Run Phase 2 Aggregator Training & Baselines (Live Demonstration)
Loads the saved Phase 1 checkpoints, performs FP32 $\to$ INT8 quantization, trains the FENS aggregators via `FedAdam` on the reserved 10% client splits ($D_{i2}$), evaluates all 6 aggregation rules from Section 3.6, prints Markdown summary tables, and generates `fig2_fens_vs_baselines.png` and `fig8_aggregator_dissection_alpha*.png` in seconds:
```bash
PYTHONPATH=src python scripts/02_run_phase2_fens_and_baselines.py --dataset cifar10 --alphas 0.01 0.05 0.1 --run-fedkd --run-fens-distill
```

#### Step 3: Custom 2-Client Skew Experiment (Generalist vs. 1–2 Class Specialist)
Tests how **Client 0** (balanced across all 10 classes) and **Client 1** (specialist with data from only 1 or 2 classes, e.g., classes `{0, 1}`) behave individually, under naive One-Shot averaging (`FedAvg-1R`, `FedEns`), and under **FENS**:
```bash
# First run (trains the 2 clients and runs Phase 2 analysis):
PYTHONPATH=src python scripts/03_run_two_client_skew_demo.py --dataset cifar10 --specialist-classes 0 1 --epochs 100

# Instant live demo reload (loads saved 2-client checkpoints and finishes in <10 seconds):
PYTHONPATH=src python scripts/03_run_two_client_skew_demo.py --dataset cifar10 --specialist-classes 0 1 --load-phase1
```
This script outputs `outputs/two_client_skew/.../two_client_skew_diagnostics.png` (4-panel figure) and `two_client_skew_report.md`, including the exact learned $2 \times 10$ per-class weight matrix $\lambda_{i, c}$.
