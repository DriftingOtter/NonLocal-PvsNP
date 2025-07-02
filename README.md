# On the Non-Local Nature of Graph Colorability

## A Computational Investigation into Emergent Constraint Entanglement

This repository contains the full research journey, code, and results for an investigation into the fundamental nature of the graph 3-colorability problem, a canonical NP-complete challenge. The project documents a methodological evolution, starting from a novel quantum-inspired "wavefunction" concept and culminating in an information-theoretic analysis that provides evidence for non-local "constraint entanglement" in complex graphs.

This work argues that the difficulty of solving certain combinatorial problems may not be merely algorithmic but may reflect fundamental properties of complex systems that resist purely local analysis.

---

## Abstract of the Research

The 3-colorability problem's difficulty has long been a subject of deep theoretical interest. This research documents an investigation into its fundamental nature. We trace a journey from exact, exponential-time solvers inspired by quantum mechanics, through a series of increasingly sophisticated polynomial-time heuristics based on hand-crafted structural features (the "hidden variable" approach), and finally to a definitive analysis using Markov Chain Monte Carlo (MCMC) sampling.

The persistent accuracy limitations of local models motivated a deeper inquiry. We designed a computational "Graph Bell Test," analogous to Bell's theorem in physics, to probe for non-local correlations. While this test did not show a violation of classical locality, a more nuanced analysis using information theory—measuring vertex entropy and mutual information—revealed strong non-local correlations in complex graphs.

We term this phenomenon **"constraint entanglement,"** where the coloring state of a vertex is intrinsically linked to the states of distant, non-adjacent vertices. This result provides a powerful, data-driven justification for the necessity of models like Graph Neural Networks (GNNs), which are uniquely suited to learning the complex, high-order, non-local correlations that define the structure of such hard combinatorial problems.

---

## Repository Structure

```
.
├── paper/                    # The final comprehensive research paper in LaTeX format.
│   ├── main.tex
│   └── images/
├── code/                     # The core Python scripts for the experiments.
│   ├── 1_mcmc_information_analysis.py
│   └── 2_computational_bell_test.py
├── requirements.txt          # A list of all required Python packages for installation.
├── LICENSE                   # MIT License for the code.
└── CC_LICENSE.txt            # Creative Commons license for the paper and written content.
```

---

## Usage Guide

This project uses Python 3. The following instructions will guide you through setting up the environment and running the key experiments.

### 1. Setup Environment

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python3 -m venv env

# Activate it (on macOS/Linux)
source env/bin/activate

# On Windows
# .\env\Scripts\activate
```

With the virtual environment activated, install all the necessary libraries from the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Running the Experiments

The `code/` directory contains the three primary Python scripts that represent the culmination of this research.

#### **Experiment 1: Information-Theoretic Analysis (Recommended)**

This script is the most insightful part of the research. It uses MCMC sampling to explore the solution space of a graph and generates the **Entropy Heatmaps** and **Mutual Information Matrices** that provide evidence for "constraint entanglement."

```bash
python code/1_mcmc_information_analysis.py
```

This script will produce several plots for a given test graph, demonstrating the quantum-like properties of the coloring problem.

#### **Experiment 2: The Computational Bell Test**

This script runs the formal "Graph Bell Test" to check for violations of a generalized Bell's inequality. **Warning:** This is a computationally intensive, `#P-hard` algorithm. It is only feasible to run on very small graphs (`n < 10`).

```bash
python code/2_computational_bell_test.py
```

Inside the script, you can change the `GRAPH_TO_TEST` variable to run the experiment on different small graphs.

## Key Findings & Visualizations

The core finding of this research is that while the 3-coloring problem may not violate a strict Bell-like inequality, it exhibits profound non-local properties. 

* **Entropy Heatmaps:** These show that the "freedom" to choose a color is not uniform across a graph. In complex graphs, some nodes become "frozen" (low entropy, blue) by distant constraints, while others remain in a "superposition" of possibilities (high entropy, red).

* **Mutual Information Matrices:** These are the "smoking gun" for non-locality. The bright, off-diagonal spots show that the color of one node can be strongly correlated with the color of another, distant node, even if they are not directly connected. This is the phenomenon we term **constraint entanglement**.

These results demonstrate that the difficulty of graph coloring arises from a complex, emergent web of global constraints, justifying the need for models like GNNs that can learn these non-local relationships.

## The Research Paper

The information-theoretic analysis, is detailed in the LaTeX document located at:

`/paper/main.tex`

This document serves as the "Opus Fundamentale" for this research project.

## License

The code in this repository is licensed under the **MIT License**. See `LICENSE` for details.
The written content, including the research paper and this README, is licensed under the **Creative Commons Attribution 4.0 International License**. See `CC_LICENSE.txt` for details.

