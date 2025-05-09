# Multi-Agent Coverage Optimization: Lloyd’s vs Simulated Annealing

This repository contains the full Python implementation of the experiments presented in the thesis:

**"Comparing Simulated Annealing and Lloyd’s Algorithm for Multi-Agent Coverage Optimization and Suggesting a Modified Hybrid Algorithm"**

The project investigates how a Modified Simulated Annealing (MSA) algorithm can outperform classical Lloyd's algorithm in optimizing coverage over non-uniform 2D domains, especially under adversarial initialization.

---

## 📁 Repository Structure

.
├── config/
│ └── settings.py # Global constants, grid and density setup
├── core/
│ ├── annealing.py # SA and Modified SA logic
│ ├── coverage_utils.py # Cost functions, centroids, Voronoi utils
│ ├── environment.py # Grid and density setup environment
│ └── lloyd.py # Lloyd's algorithm implementation
├── results/
│ └── [scenario folders] # Saved results for all experiments (images + txt)
├── visual/
│ └── plotter.py # Image generation and convergence graphing
├── main.py # Entry point to run experiments
├── venv/ # Virtual environment (optional, not included)
└── README.md 

---

## ⚙️ How to Run

1. **Install dependencies (if not using venv):**

```bash
pip install numpy matplotlib scipy
To run an experiment:
Modify scenario settings inside main.py or specific files in config/ and results/.

python main.py
Output:
Results (convergence graphs, Voronoi partitions, heatmaps, etc.) are stored in results/<scenario_name>.

🧪 Scenarios Implemented
Scenario ID	Description
1	Uniform density (base case for Lloyd’s)
2	Weak Gaussian center, shows SA slower but better
3	Agents right, density left – SA variants vs Lloyd's
4	One agent forward – Lloyd's performs decently
5	Random agent initialization
6	Standard vs Modified SA under same init
7	Alpha tuning – low alpha converges fast but suboptimal

Each folder in results/ is named accordingly to explain its content.

📊 Algorithm Overview
Lloyd’s Algorithm: Deterministic centroid descent with fast convergence but highly sensitive to initialization.

Simulated Annealing (SA): Stochastic global optimizer; robust to local minima but slow convergence.

Modified SA: Hybrid approach combining centroid guidance, temperature-dependent exploration, and step decay.

📌 Parameters
Key adjustable parameters:

Initial temperature (T0)

Cooling rate (alpha)

Step size decay

Centroid move probability (β(T))

These can be configured in core/annealing.py and config/settings.py.

📎 Citation / Thesis Link
Travis, Bachelor Thesis in Industrial Engineering & Management, University of Groningen, 2025.
Title: "Comparing Simulated Annealing and Lloyd’s Algorithm for Multi-Agent Coverage Optimization and Suggesting a Modified Hybrid Algorithm"
```
