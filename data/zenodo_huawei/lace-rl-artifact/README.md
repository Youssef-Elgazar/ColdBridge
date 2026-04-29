# LACE-RL (CCGrid'26 Artifact) — Demo

This repository contains the **artifact** for the paper "Green or Fast? Learning to Balance Cold Starts and Idle Carbon in Serverless Computing" accepted at CCGrid 2026.

The artifact provides:
- a **DQN agent ready for inference** (pretrained checkpoint),
- a **trace-driven simulator** implementing the paper’s metrics accounting,
- a **demo script** that runs evaluation and outputs metrics,
- the **demo datasets**

---

## Repository Structure
.
├── lace_rl/
│   ├── agent/
│   │   └── dqn_agent.py # DQN definition
│   ├── sim/
│   │   └── trace_simulator.py # trace-driven simulator
│   └── utils/
│       └── io.py
├── scripts/
│   └── run_demo.py # demo entrypoint
├── configs/
│   └── demo.json # config
├── checkpoints/
│   └── trained_agent.pth # pretrained model checkpoint
├── data/
│   └── demo/ # sampled data
│   ├── invocations.pkl
│   ├── green_trace.pkl
│   ├── network_latency_map.pkl # optional
│   └── cs_latency_dict.pkl # optional
├── results/ # demo output folder
├── requirements.txt
└── README.md

---
## Quick Start (Demo)

### 1) Environment Setup

Requires **Python 3.10+**.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # minimal dependencies listed
```

### 2) Run the Demo

```bash
PYTHONPATH=. python scripts/run_demo.py --config configs/demo.json
```
### 3) Output

The demo prints runtime and writes:
+ results/metrics.json

The JSON includes the main metrics defined in the paper (please refer to the paper for more details).

## Notes and Limitations

+ The demo uses a fixed pretrained checkpoint.

+ Artifact evaluation focuses on reproducing behaviors/trends-only.

+ Third-party code/data (if any) are not claimed as our contribution and should be used under their original terms.

+ The demo inputs included in this artifact are a **processed subset** of the data used in our paper, derived from the Huawei Public Cloud trace (Joosen et al., EuroSys 2025) and carbon-intensity time series from Electricity Maps. The full original datasets are not included in this repository; please obtain them from the respective providers and follow their terms of use.

## Citation

If you use this artifact, please cite our paper: Bowen Sun, Christos Antonopoulos, Evgenia Smirni, Bin Ren, Nikolaos Bellas, and Spyros Lalis, "Green or Fast? Learning to Balance Cold Starts and Idle Carbon in Serverless Computing," in Proceedings of the 26th IEEE International Symposium on Cluster, Cloud, and Internet Computing (CCGrid'26).

## Acknowledgement

This work has been supported in part by the Horizon Europe research and innovation programme of the European Union, under grant agreement no 101092912, project MLSysOps and the U.S. National Science Foundation (NSF) grants \#2402942 and \#2403088.