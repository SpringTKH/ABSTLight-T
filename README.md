# ABSTLight-T: OOD-Resilient Multi-Agent Traffic Signal Control

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![SUMO](https://img.shields.io/badge/Eclipse%20SUMO-Required-orange)
![License](https://img.shields.io/badge/License-MIT-green)

ABSTLight-T is a thesis-driven, multi-agent reinforcement learning framework for adaptive traffic signal control under out-of-distribution (OOD) stressors, including rain degradation and lane-blocking accidents. The system is built on a Kuala Lumpur OpenStreetMap (OSM) topology, uses an 82-dimensional spatiotemporal state tensor (80-D stacked traffic state + 2-D context tags), and integrates a graph-masked multi-head attention module (GCN-MHA) for topology-aware coordination across intersections.

> A traffic-signal control system that keeps working when things go wrong — heavy rain, accidents, blocked lanes — by combining graph-based spatial reasoning with temporal traffic patterns. Achieves up to 49% less delay than conventional traffic light systems under real-world conditions.

## Key Engineering Features

- **Deadlock-prevention reward shaping:** Accident-time penalties are localized to controllable open lanes, explicitly isolating uncontrollable physical blockage effects from policy gradients.
- **RELight conservatism mechanism:** Ensemble Q-heads with M-subset minimum target selection and threshold-based subset resampling reduce optimistic value bias.
- **Throughput defined as destination arrivals:** Reward and evaluation throughput are based on destination arrival rates (`arrived_vehicles / simulation_seconds`), not naive intersection passage counts.
- **OOD scenario conditioning:** Explicit scenario/context handling for sunny, rainy, and accident regimes with dynamic activation support.
- **Statistical protocol over fixed seeds:** Built-in multi-seed runner evaluates all baseline/agent modes over 5 seeds (`42, 100, 1234, 2026, 8888`) for reproducible summary statistics.

## Repository Structure

```text
ABSTLight/
├── agent/                          # Agent logic + RELight update rule
│   └── abst_light_agent.py
├── environment/                    # SUMO/TraCI envs for base + OOD mixed modes
│   ├── traffic_env.py
│   └── traffic_env_mixed.py
├── memory/                         # Experience replay implementation
│   └── replay_buffer.py
├── model_components/               # Neural modules used by ABSTLight
│   ├── observation_embedding.py
│   ├── gcn_mha.py
│   └── q_value_head.py
├── sumo_files/                     # SUMO network, route, cfg, and TLS assets
├── sumo_tools/                     # Data/scenario generation scripts
│   ├── gen_routes.py
│   ├── generate_rainy_xml.py
│   ├── gen_accidents_routes.py
│   └── run_pipeline.py
├── utils/
│   └── metrics_tracker.py          # Training/evaluation metric utilities
├── docs/                           # Thesis and architecture documentation
│   ├── ABSTLIGHT_TECHNICAL_ARCHITECTURE_SPEC.md
│   └── CHAPTER_3_METHODOLOGY.md
├── model/                          # Saved base checkpoints (e.g., episode_1000/)
├── model_finetune/                 # Fine-tuned model checkpoints
├── logs/                           # Training and fine-tuning JSON logs
├── results/                        # Structured evaluation outputs/summaries
│   ├── accident/
│   │   ├── acc1/
│   │   ├── acc2/
│   │   ├── acc3/
│   │   └── acc4/
│   ├── normal/
│   ├── peak-hours/
│   ├── rainy/
│   │   ├── moderate_rain/
│   │   └── heavy_rain/
├── requirements.txt                # Python dependencies
├── config.py                       # Core configuration (ABSTConfig)
├── train.py                        # Base ABSTLight training/testing entrypoint
├── train_finetune.py               # 3-in-1 fine-tuning pipeline (82-D)
├── test.py                         # Unified evaluation runner (models 1/2/3/4)
├── run_seed_suite.py               # 5-seed statistical evaluation driver
└── weight_converter.py             # Embedding conversion for fine-tuning bootstrap
```

Detailed architecture notes and publication-facing technical documentation are available in the `docs/` directory.

## Getting Started

### 1) Prerequisites

- Install **Eclipse SUMO** (including TraCI tools).
- Set the `SUMO_HOME` environment variable so Python can import `traci` and `sumolib`.

Windows (PowerShell):

```powershell
$env:SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
```

Linux/macOS (bash):

```bash
export SUMO_HOME=/usr/share/sumo
```

### 2) Clone and install dependencies

```bash
git clone <https://github.com/SpringTKH/ABSTLight-T.git>
cd ABSTLight
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Base Training (`train.py`)

```bash
python train.py
```

Useful options:

```bash
python train.py --gui
python train.py --episodes 100
python train.py --route-file sumo_files/osm.moderate.rou.xml
```

### Fine-Tuning (`train_finetune.py`)

First run requires base checkpoint assets (including `model/episode_1000/finetune_ready_embedding.pth` from `weight_converter.py`):

```bash
python train_finetune.py
```

Useful options:

```bash
python train_finetune.py --gui
python train_finetune.py --episodes 300
python train_finetune.py --epsilon 0.2
```

### Evaluation (`test.py`)

Model modes:

- `--model 1`: SUMO default TLS baseline
- `--model 2`: ABSTLight base model
- `--model 3`: SUMO fixed-time baseline
- `--model 4`: ABSTLight fine-tuned model

Examples:

```bash
python test.py --model 2 --route-file sumo_files/osm.moderate.rou.xml --seed 42
python test.py --model 2 --rainy --route-file sumo_files/osm.rain_m.rou.xml --seed 100
python test.py --model 4 --accident --route-file sumo_files/osm.acc1.rou.xml --seed 1234
python test.py --model 1 --gui --route-file sumo_files/osm.peak.rou.xml
```

Notes:

- `test.py` enforces single-episode evaluation per invocation.
- Results are saved to structured JSON files under `results/`.

### Five-Seed Statistical Suite

Run all four model modes across the fixed 5-seed protocol for a single route:

```bash
python run_seed_suite.py --route-file sumo_files/osm.moderate.rou.xml
```

This produces aggregated summary statistics in `results/.../final_summary_*.json`.

## Citation

If you use ABSTLight-T in your research, please cite:

```bibtex
@misc{ho2026abstlightt,
  title        = {ABSTLight-T: OOD-Resilient Multi-Agent Traffic Signal Control},
  author       = {Ho, Tee Ken},
  year         = {2026},
  publisher    = {Xiamen University Malaysia},
  note         = {Final Year Project Thesis},
  howpublished = {GitHub repository}
}
```

## Results Highlight

Evaluated via a 5-seed protocol across normal, peak-hour, rainy (OOD), and 
accident (OOD) traffic scenarios on a real-world Kuala Lumpur road network.

| Scenario | Metric | ABSTLight-T | Best Baseline (Actuated) | Improvement |
|---|---|---|---|---|
| Peak-Hours | Time Loss | 152.89s | 208.31s | ~27% ↓ |
| Peak-Hours | Max Queue Length | 35.60 veh | 60.20 veh | ~41% ↓ |
| Accident (OOD, Jalan Petaling) | Time Loss (FT-model) | 73.45s | 105.42s | ~30% ↓ |
| Accident (OOD) | Vehicles Reaching Destination | 100% (2,118/2,118) | — | Zero deadlocks |

All improvements validated via independent-samples t-tests (p < 0.05) across 5 random seeds.

## License

This repository is intended for open-source release under the MIT License.
