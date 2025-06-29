# FILE: 1_mcmc_information_analysis.py
# DESC: Implements the MCMC sampling and information-theoretic analysis from 
#       Section 4 of the paper "On the Non-Local Nature of Graph Colorability."
#       This code generates the data needed to produce Figure 1, providing evidence
#       for "constraint entanglement" in complex graphs.

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import Counter, defaultdict
from itertools import product

# --- Configuration ---
# Corresponds to the parameters described in Section 4.1
COLORS = [0, 1, 2]
BETA = 5.0          # Inverse temperature for MCMC sampling
NUM_SAMPLES = 30000 # Number of samples to generate

def get_benchmark_graph(name):
    """
    Returns a networkx graph object for key benchmark graphs from the paper.
    """
    if name.lower() == "cycle c5":
        # A simple, "classical" graph as per Section 4.2
        return nx.cycle_graph(5)
    elif name.lower() == "groetzsch":
        # A complex, "entangled" graph as per Section 4.2
        G = nx.Graph()
        G.add_nodes_from(range(11))
        G.add_edges_from([(0,1), (0,2), (0,3), (0,4), (0,5), (1,6), (1,8), (1,10), 
                          (2,6), (2,7), (2,9), (3,7), (3,8), (3,10), (4,6), (4,9), 
                          (4,10), (5,7), (5,8), (5,9)])
        return G
    else:
        raise ValueError(f"Unknown graph name: {name}")

# --- Core MCMC and Statistical Mechanics Logic (Section 4.1) ---

def coloring_energy(graph, coloring):
    """
    Calculates the "energy" of a given coloring, defined as the number of
    monochromatic edges (constraint violations).
    E(C) in the paper's notation.
    """
    energy = 0
    for u, v in graph.edges():
        if coloring.get(u) == coloring.get(v):
            energy += 1
    return energy

def mcmc_sampler(graph, num_samples, beta, k=3):
    """
    Samples from the Boltzmann distribution of graph colorings P(C) ~ exp(-beta * E(C))
    using the Metropolis-Hastings algorithm, as detailed in Algorithm 1 of the paper.
    """
    print(f"Running MCMC sampler for {num_samples} steps with beta={beta}...")
    nodes = list(graph.nodes())
    # Step 2: Initialize a random coloring
    current_coloring = {v: random.randint(0, k-1) for v in nodes}
    samples = []

    # Step 4: Loop for N_s samples
    for _ in range(num_samples):
        # Step 6: Select a random vertex
        v_to_flip = random.choice(nodes)
        
        # Step 7: Select a new random color
        old_color = current_coloring[v_to_flip]
        possible_new_colors = [c for c in range(k) if c != old_color]
        new_color = random.choice(possible_new_colors)
        
        # Step 9: Compute energy change delta_E
        old_energy = coloring_energy(graph, current_coloring)
        current_coloring[v_to_flip] = new_color
        new_energy = coloring_energy(graph, current_coloring)
        delta_E = new_energy - old_energy

        # Step 10: Metropolis-Hastings acceptance criteria
        if delta_E <= 0 or random.random() < math.exp(-beta * delta_E):
            # Accept the new state (Step 11)
            pass  
        else:
            # Reject and keep the old state (Step 13)
            current_coloring[v_to_flip] = old_color
            
        # Step 14: Add the current state to our list of samples
        samples.append(current_coloring.copy())

    print("Sampling complete.")
    return samples

def compute_vertex_entropy(samples, graph, k=3):
    """
    Computes the Shannon entropy for each vertex based on its marginal
    probability distribution of colors, as per the formula for H(v) in Section 4.1.
    A high entropy indicates a "superposition" of states.
    """
    marginals = {v: Counter() for v in graph.nodes()}
    for sample in samples:
        for v in graph.nodes():
            marginals[v][sample[v]] += 1

    entropies = {}
    total_samples = len(samples)
    for v, counts in marginals.items():
        entropy = 0.0
        for color in range(k):
            p = counts[color] / total_samples
            if p > 0:
                entropy -= p * math.log2(p)
        entropies[v] = entropy
    return entropies

def compute_mutual_information(samples, graph, k=3):
    """
    Computes the mutual information I(u;v) for every pair of vertices,
    quantifying the correlation between them regardless of distance.
    This implements the formula for I(u;v) from Section 4.1.
    """
    nodes = list(graph.nodes())
    num_nodes = len(nodes)
    node_map = {node: i for i, node in enumerate(nodes)}
    mi_matrix = np.zeros((num_nodes, num_nodes))
    total_samples = len(samples)

    # Compute marginal probabilities P(c)
    p_marginal = [defaultdict(float) for _ in range(num_nodes)]
    for sample in samples:
        for i, node in enumerate(nodes):
            p_marginal[i][sample[node]] += 1/total_samples
    
    # Compute joint probabilities P(cu, cv) and MI
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            u, v = nodes[i], nodes[j]
            if i == j:
                # MI of a variable with itself is its entropy
                entropy = 0
                for p in p_marginal[i].values():
                    if p > 0: entropy -= p * math.log2(p)
                mi_matrix[i, j] = entropy
                continue

            p_joint = defaultdict(float)
            for sample in samples:
                p_joint[(sample[u], sample[v])] += 1/total_samples
            
            mi = 0.0
            for (cu, cv), p_uv in p_joint.items():
                if p_uv > 0:
                    p_u = p_marginal[i][cu]
                    p_v = p_marginal[j][cv]
                    mi += p_uv * math.log2(p_uv / (p_u * p_v))
            
            mi_matrix[i, j] = mi_matrix[j, i] = mi
            
    return mi_matrix, nodes

# --- Visualization (to generate plots similar to Figure 1) ---

def visualize_results(graph, graph_name, entropies, mi_matrix, mi_nodes):
    """
    Generates and displays the Entropy Heatmap and Mutual Information Matrix.
    """
    print(f"Generating visualizations for {graph_name}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Information-Theoretic Analysis of: {graph_name}", fontsize=16)

    # 1. Entropy Heatmap (like Fig 1a, 1c)
    pos = nx.spring_layout(graph, seed=42)
    node_colors = [entropies[v] for v in graph.nodes()]
    vmax = math.log2(len(COLORS)) # Theoretical maximum entropy
    nodes = nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap=plt.cm.coolwarm, 
                                 vmin=0, vmax=vmax, node_size=500, ax=ax1)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=ax1)
    nx.draw_networkx_labels(graph, pos, font_color='white', ax=ax1)
    ax1.set_title("Vertex Entropy Heatmap")
    fig.colorbar(nodes, ax=ax1, orientation='vertical', label="Shannon Entropy")

    # 2. Mutual Information Matrix (like Fig 1b, 1d)
    im = ax2.imshow(mi_matrix, cmap='viridis')
    ax2.set_title("Mutual Information Between Nodes")
    ax2.set_xticks(np.arange(len(mi_nodes)))
    ax2.set_yticks(np.arange(len(mi_nodes)))
    ax2.set_xticklabels(mi_nodes)
    ax2.set_yticklabels(mi_nodes)
    fig.colorbar(im, ax=ax2, orientation='vertical', label="Mutual Information")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    # --- Run Analysis on Benchmark Graphs from the Paper ---
    # This analysis directly reproduces the findings of Section 4.2
    
    # 1. The "Classical" Graph: Cycle C5
    G_c5 = get_benchmark_graph("Cycle C5")
    samples_c5 = mcmc_sampler(G_c5, num_samples=NUM_SAMPLES, beta=BETA)
    entropies_c5 = compute_vertex_entropy(samples_c5, G_c5)
    mi_matrix_c5, mi_nodes_c5 = compute_mutual_information(samples_c5, G_c5)
    visualize_results(G_c5, "Cycle C₅ (Classical)", entropies_c5, mi_matrix_c5, mi_nodes_c5)

    # 2. The "Entangled" Graph: Grötzsch
    G_groetzsch = get_benchmark_graph("Groetzsch")
    samples_groetzsch = mcmc_sampler(G_groetzsch, num_samples=NUM_SAMPLES, beta=BETA)
    entropies_groetzsch = compute_vertex_entropy(samples_groetzsch, G_groetzsch)
    mi_matrix_groetzsch, mi_nodes_groetzsch = compute_mutual_information(samples_groetzsch, G_groetzsch)
    visualize_results(G_groetzsch, "Grötzsch Graph (Entangled)", entropies_groetzsch, mi_matrix_groetzsch, mi_nodes_groetzsch)
