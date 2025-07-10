# On the Non-Local Nature of Graph Colorability

**A Computational Investigation into Emergent Constraint Entanglement**

*Daksh Kaul - July 10, 2025*

This repository contains the full research journey, code, and results for an investigation into the fundamental nature of the graph 3-colorability problem, a canonical NP-complete challenge. The project documents a methodological evolution, starting from a novel quantum-inspired "wavefunction" concept and culminating in an information-theoretic analysis that provides evidence for non-local "constraint entanglement" in complex graphs. This work argues that the difficulty of solving certain combinatorial problems may not be merely algorithmic but may reflect fundamental properties of complex systems that resist purely local analysis.

## Abstract of the Research

The 3-colorability of a graph is a canonical NP-complete problem whose computational hardness remains a topic of deep theoretical interest. This paper presents a methodological investigation that begins with a quantum-inspired "wavefunction" model and culminates in a statistical mechanics-based simulation revealing non-local properties in the graph coloring solution space. The persistent limitations of local, "hidden variable" models prompted a shift toward a global perspective on graph constraints. Through an information-theoretic lens, we uncover strong non-local correlations using measures of vertex entropy and mutual information. These findings provide compelling evidence for a phenomenon we term **constraint entanglement**, in which the coloring state of a vertex is intrinsically correlated with the states of distant, non-adjacent vertices. To characterize this effect, we categorize graphs along a "non-locality spectrum" using five representative archetypes: classical (Cycle Graph), jammed (Complete Graph), structured (Circulant Graph), community (Wheel Graph), and entangled (Erdős-Rényi Graph). These results offer a data-driven rationale for the use of Graph Neural Networks (GNNs), which are uniquely equipped to learn and model the complex, high-order, non-local correlations that underlie the structure of hard combinatorial problems.

## Repository Structure

```
.
├── paper/                              # The final comprehensive research paper in LaTeX format
│   ├── main.tex
│   └── images/
├── code/                               # The core Python scripts for the experiments
│   ├── 1_mcmc_information_analysis.py
│   └── 2_graph_archetype_analysis.py
├── requirements.txt                    # A list of all required Python packages for installation
├── LICENSE                            # MIT License for the code
├── CC_LICENSE.txt                     # Creative Commons license for the paper and written content
└── README.md                          # This file
```

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

With the virtual environment activated, install all the necessary libraries from the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

### 2. Running the Experiments

The `code/` directory contains the two primary Python scripts that represent the culmination of this research.

#### Experiment 1: Information-Theoretic Analysis (Recommended)

This script uses MCMC sampling to explore the solution space of a graph and generates the Entropy Heatmaps and Mutual Information Matrices that provide evidence for "constraint entanglement".

```bash
python code/1_mcmc_information_analysis.py
```

This script will produce several plots for a given test graph, demonstrating the quantum-like properties of the coloring problem.

#### Experiment 2: Graph Archetype Analysis and Visualization

This script analyzes various graph archetypes (e.g., Cycle, Complete, Circulant, Wheel, Erdős-Rényi graphs) by generating them, performing MCMC sampling, and then computing and visualizing their vertex entropy and mutual information matrices. It demonstrates the spectrum of complexity from "classical" local systems to "entangled" non-local systems, as discussed in the paper.

```bash
python code/2_graph_archetype_analysis.py
```

This script will output various statistics and generate the visualizations presented in the research paper for each canonical graph archetype.

## Key Findings & Visualizations

The core finding of this research is that while the 3-coloring problem may not violate a strict Bell-like inequality, it exhibits profound non-local properties. The analysis reveals a clear "non-locality spectrum" that categorizes graphs into five representative archetypes:

1. **Classical (Cycle Graph)**: Uniformly high entropy with strictly local correlations
2. **Jammed (Complete Graph)**: Uniformly low entropy with global impossibility constraints  
3. **Structured (Circulant Graph)**: Patterned entropy with orderly constraint propagation
4. **Community (Wheel Graph)**: Hierarchical entropy with hub-mediated correlations
5. **Entangled (Erdős-Rényi Graph)**: Rich, non-uniform entropy with long-range correlations

These findings are primarily supported by the visualizations generated by `2_graph_archetype_analysis.py`:

- **Entropy Heatmaps**: These show that the "freedom" to choose a color is not uniform across a graph. In complex graphs, some nodes become "frozen" (low entropy) by intricate constraints, while others remain in a high-entropy "superposition" of possibilities.

- **Mutual Information Matrices**: These are the "smoking gun" for non-locality. The bright, off-diagonal spots show significant correlations between distant, non-adjacent nodes. This is the phenomenon we term **constraint entanglement**, implying that assigning a color to one vertex can significantly reduce the uncertainty about the color of a far-off, seemingly unrelated vertex, demonstrating a truly global and non-local dependency structure.

The results demonstrate that the difficulty of graph coloring arises not from simple violation of locality, but from an incredibly high-order, complex web of local dependencies that emulate non-local behavior, justifying the need for models like GNNs that can learn these non-local relationships.

## The Research Paper

The information-theoretic analysis is detailed in the LaTeX document located at:

```
/paper/main.tex
```

This document serves as the "Opus Fundamentale" for this research project.

## License

The code in this repository is licensed under the MIT License. See `LICENSE` for details.

The written content, including the research paper and this README, is licensed under the Creative Commons Attribution 4.0 International License. See `CC_LICENSE.txt` for details.
