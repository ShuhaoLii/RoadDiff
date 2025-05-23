# Fine-Grained Traffic Inference from Road to Lane via Spatio-Temporal Graph Node Generation

This repository contains the PyTorch implementation of **RoadDiff**, a two-stage framework for fine-grained traffic inference. RoadDiff captures the spatiotemporal dependencies between roads and lanes and leverages a constraint-aware diffusion module to generate accurate lane-level traffic states under uncertainty.

## 💡 Motivation

- **Fine-grained traffic inference** reduces reliance on dense lane-level sensors or complex tracking systems, enabling downstream applications such as lane change guidance and adaptive signal control.
- The problem of **spatio-temporal graph node generation** has broad research significance in domains such as traffic systems, social networks, weather forecasting, and financial modeling.

## 🔍 Framework Overview

![RoadDiff Overview](./fig/KDD25_RoadDiff.png "Overview of RoadDiff")

- **Spatiotemporal Modeling:** We treat road/lane segments at different times as distinct nodes and use a simple attention-based network to model spatiotemporal dependencies across nodes.
- **Lane Diffusion Module:** A diffusion-based refinement process models noise and uncertainty while enforcing physical constraints (e.g., flow conservation and speed consistency) to improve the fidelity of lane-level outputs.

## ⚙️ Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

We recommend Python 3.8 and PyTorch 1.11.0 to match our development environment.

## 📁 Datasets

Place all datasets in the `Datasets/` directory. If using your own data, please ensure it follows the same format, including correct definitions of traffic state types and the number of lanes per road segment (`road_lane_count`).

## 🚀 Run Instructions

Before running the model, ensure that the hyperparameters are appropriately configured in `args.py`. The default settings used in our experiments are:

- **Learning Rate:** 1e-4  
- **Batch Size:** 64  
- **Training Iterations:** 1000  
- **Diffusion Steps:** 10  
- **Early Stopping:** Enabled (starting from epoch 20, learning rate is halved every 10 epochs)  
- **Optimizer:** Adam  

To start training:

```bash
python main.py
```

Make sure that the traffic state configurations and lane topology of your dataset match the model's expectations.

---

📬 If you have any questions or suggestions, feel free to contact us or open an issue. Contributions are welcome!