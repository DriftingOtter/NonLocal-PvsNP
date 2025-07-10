# FILE: 2_graph_archetype_analysis.py

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import time
import multiprocessing
from functools import partial
from collections import Counter, defaultdict
import random
import math
import warnings

warnings.filterwarnings("ignore")

# --- Core MCMC, MI Matrix, and Fingerprint Calculation ---

def mcmc_sampler(graph, num_samples=30000, beta=5.0, burn_in=1000, thin=5):
    """High-quality MCMC sampler with proper error handling."""
    nodes = list(graph.nodes())
    k = 3
    if not nodes or len(nodes) == 0:
        return []
    
    # Ensure graph has edges
    if graph.number_of_edges() == 0:
        return []
    
    coloring = {v: random.randint(0, k-1) for v in nodes}
    
    # Burn-in period
    for _ in range(burn_in):
        v_flip = random.choice(nodes)
        old_color = coloring[v_flip]
        new_color = random.choice([c for c in range(k) if c != old_color])
        
        # Calculate energy change properly
        old_conflicts = sum(1 for n in graph.neighbors(v_flip) if coloring.get(n) == old_color)
        new_conflicts = sum(1 for n in graph.neighbors(v_flip) if coloring.get(n) == new_color)
        delta_E = new_conflicts - old_conflicts
        
        if delta_E <= 0 or random.random() < math.exp(-beta * delta_E):
            coloring[v_flip] = new_color
    
    # Sampling with thinning
    samples = []
    for i in range(num_samples * thin):
        v_flip = random.choice(nodes)
        old_color = coloring[v_flip]
        new_color = random.choice([c for c in range(k) if c != old_color])
        
        old_conflicts = sum(1 for n in graph.neighbors(v_flip) if coloring.get(n) == old_color)
        new_conflicts = sum(1 for n in graph.neighbors(v_flip) if coloring.get(n) == new_color)
        delta_E = new_conflicts - old_conflicts
        
        if delta_E <= 0 or random.random() < math.exp(-beta * delta_E):
            coloring[v_flip] = new_color
        if i % thin == 0:
            samples.append(coloring.copy())
    return samples

def compute_mutual_information_matrix(samples, graph):
    """Computes the full n x n mutual information matrix with error handling."""
    nodes = list(graph.nodes())
    num_nodes = len(nodes)
    if num_nodes == 0 or not samples:
        return np.array([])
    
    mi_matrix = np.zeros((num_nodes, num_nodes))
    total_samples = len(samples)
    
    # Calculate marginal probabilities
    p_marginal = [defaultdict(float) for _ in range(num_nodes)]
    for sample in samples:
        for i, node in enumerate(nodes):
            p_marginal[i][sample[node]] += 1/total_samples
    
    # Calculate mutual information matrix
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            u, v = nodes[i], nodes[j]
            if i == j:
                # Diagonal elements are entropy
                entropy = sum(-p * math.log2(p) for p in p_marginal[i].values() if p > 0)
                mi_matrix[i, j] = entropy
                continue
            
            # Calculate joint probabilities
            p_joint = defaultdict(float)
            for sample in samples:
                p_joint[(sample[u], sample[v])] += 1/total_samples
            
            # Calculate mutual information
            mi = 0.0
            for (cu, cv), p_uv in p_joint.items():
                if p_uv > 0 and p_marginal[i][cu] > 0 and p_marginal[j][cv] > 0:
                    mi += p_uv * math.log2(p_uv / (p_marginal[i][cu] * p_marginal[j][cv]))
            
            mi_matrix[i, j] = mi_matrix[j, i] = max(0, mi)  # Ensure non-negative
    
    return mi_matrix

def calculate_matrix_fingerprint(matrix):
    """Calculates a sophisticated feature vector with proper error handling."""
    if matrix.size < 4:
        return np.zeros(10)
    
    # Handle NaN and infinite values
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    diag = np.diag(matrix)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    off_diag = matrix[mask]
    
    # Enhanced feature set for better discrimination
    features = [
        np.mean(diag) if len(diag) > 0 else 0,
        np.std(diag) if len(diag) > 1 else 0,
        np.mean(off_diag) if len(off_diag) > 0 else 0,
        np.std(off_diag) if len(off_diag) > 1 else 0,
        np.max(off_diag) if len(off_diag) > 0 else 0,
        np.min(off_diag) if len(off_diag) > 0 else 0,
        np.median(off_diag) if len(off_diag) > 0 else 0,
        np.sum(off_diag > 0.1) if len(off_diag) > 0 else 0,
        np.sum(off_diag < 0.01) / len(off_diag) if len(off_diag) > 0 else 1,
        np.sum(off_diag > np.mean(off_diag)) / len(off_diag) if len(off_diag) > 0 else 0
    ]
    
    # Ensure all features are finite
    features = [f if np.isfinite(f) else 0.0 for f in features]
    
    return np.array(features)

# --- Canonical Graph Generation ---

def generate_canonical_graphs():
    """Generate canonical graphs as described in the taxonomy paper."""
    canonical_graphs = {}
    
    # Category 1&2: Entangled Archetypes (ErdÅs-RÃ©nyi-like)
    # Mid-to-high density random graphs
    canonical_graphs['Entangled (ER-like)'] = nx.erdos_renyi_graph(12, 0.4, seed=42)
    
    # Category 3: Classical Archetype (Path-like)
    # Cycle, Path, Tree graphs - locally constrained
    canonical_graphs['Classical (Path-like)'] = nx.cycle_graph(10)
    
    # Category 4: Jammed Archetype (Regular/Dense)
    # Small, high-density regular graphs - over-constrained
    canonical_graphs['Jammed (Regular/Dense)'] = nx.complete_graph(6)
    
    # Category 5: Structured Archetype (Low-Degree Regular)
    # Low-degree regular graphs with structured patterns
    canonical_graphs['Structured (Low-Degree Regular)'] = nx.circulant_graph(8, [1, 2])
    
    # Category 6: Community Archetype (High-Density ER/Wheel-like)
    # Decomposable structure with tight communities
    canonical_graphs['Community (Wheel-like)'] = nx.wheel_graph(8)
    
    return canonical_graphs

# --- Enhanced Archetype Analysis ---

def classify_archetype(fingerprint, canonical_fingerprints):
    """Classify a graph fingerprint to the closest canonical archetype."""
    distances = {}
    for name, canonical_fp in canonical_fingerprints.items():
        dist = distance.euclidean(fingerprint, canonical_fp)
        distances[name] = dist
    
    # Return the closest archetype
    return min(distances, key=distances.get)

def create_vertex_entropy_visualization(mi_matrix, graph, category_name):
    """Create a vertex entropy visualization showing the graph with entropy-colored nodes."""
    if mi_matrix.size == 0:
        return None
    
    # Extract diagonal elements (vertex entropies)
    vertex_entropies = np.diag(mi_matrix)
    
    # Create layout
    pos = nx.spring_layout(graph, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw nodes with entropy-based coloring
    nodes = nx.draw_networkx_nodes(graph, pos, ax=ax,
                                  node_color=vertex_entropies,
                                  node_size=600,
                                  cmap='viridis',
                                  alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.5, edge_color='gray', width=2)
    
    # Add labels
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=12, font_color='white', font_weight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(nodes, ax=ax, label='Vertex Entropy', shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    
    # Set labels only - title will be explained in paper
    ax.set_xlabel('Node Index', fontsize=14)
    ax.set_ylabel('Node Index', fontsize=14)
    
    # Add graph statistics as text
    stats_text = f"Nodes: {graph.number_of_nodes()}\nEdges: {graph.number_of_edges()}\nDensity: {nx.density(graph):.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.axis('off')
    
    return fig

def create_mi_matrix_visualization(mi_matrix, category_name):
    """Create a mutual information matrix visualization."""
    if mi_matrix.size == 0:
        return None
    
    # Clean the matrix
    mi_clean = np.nan_to_num(mi_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(mi_clean, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mutual Information', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Remove title - will be explained in paper
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add MI statistics as text
    diag_mean = np.mean(np.diag(mi_clean))
    off_diag = mi_clean[~np.eye(mi_clean.shape[0], dtype=bool)]
    mi_mean = np.mean(off_diag) if len(off_diag) > 0 else 0
    mi_max = np.max(off_diag) if len(off_diag) > 0 else 0
    
    stats_text = f"Mean Vertex Entropy: {diag_mean:.3f}\nMean MI: {mi_mean:.3f}\nMax MI: {mi_max:.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    return fig

def worker_generate_graph_data(params):
    """Worker function to generate a graph, its MI matrix, and its fingerprint."""
    try:
        n, m, seed_offset = params
        seed = int(time.time() * 1000) + random.randint(0, 10000) + seed_offset
        
        G = nx.gnm_random_graph(n, m, seed=seed)
        
        # Ensure graph has edges
        if G.number_of_edges() == 0:
            # Add minimum edges to make it connected
            nodes = list(G.nodes())
            for i in range(len(nodes) - 1):
                G.add_edge(nodes[i], nodes[i + 1])
        
        samples = mcmc_sampler(G)
        if not samples:
            return None
            
        mi_matrix = compute_mutual_information_matrix(samples, G)
        if mi_matrix.size == 0:
            return None
            
        fingerprint = calculate_matrix_fingerprint(mi_matrix)
        
        # Validate fingerprint
        if not np.all(np.isfinite(fingerprint)):
            return None
        
        return {'graph': G, 'mi_matrix': mi_matrix, 'fingerprint': fingerprint}
        
    except Exception as e:
        return None

def run_taxonomy_analysis():
    print("--- Taxonomy-Based Archetype Analysis ---")
    print("Based on: A Taxonomy of Computational Complexity")
    start_time = time.time()
    
    # Generate canonical graphs and their fingerprints
    print("Generating canonical graph archetypes...")
    canonical_graphs = generate_canonical_graphs()
    canonical_fingerprints = {}
    
    for name, graph in canonical_graphs.items():
        print(f"  Processing {name}...")
        samples = mcmc_sampler(graph)
        if samples:
            mi_matrix = compute_mutual_information_matrix(samples, graph)
            if mi_matrix.size > 0:
                fingerprint = calculate_matrix_fingerprint(mi_matrix)
                canonical_fingerprints[name] = fingerprint
                canonical_graphs[name] = {'graph': graph, 'mi_matrix': mi_matrix, 'fingerprint': fingerprint}
    
    print(f"Canonical archetypes processed: {len(canonical_fingerprints)}")
    
    # Display individual archetype analysis
    print("\n" + "="*80)
    print("INDIVIDUAL ARCHETYPE ANALYSIS")
    print("="*80)
    
    for name, data in canonical_graphs.items():
        if 'mi_matrix' not in data:
            continue
            
        print(f"\n--- {name} ---")
        graph = data['graph']
        mi_matrix = data['mi_matrix']
        
        # Print graph properties
        print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        print(f"Density: {nx.density(graph):.3f}")
        print(f"Clustering: {nx.average_clustering(graph):.3f}")
        
        # Print MI matrix properties
        if mi_matrix.size > 0:
            diag_mean = np.mean(np.diag(mi_matrix))
            off_diag = mi_matrix[~np.eye(mi_matrix.shape[0], dtype=bool)]
            mi_mean = np.mean(off_diag) if len(off_diag) > 0 else 0
            mi_max = np.max(off_diag) if len(off_diag) > 0 else 0
            
            print(f"Mean Vertex Entropy: {diag_mean:.3f}")
            print(f"Mean MI: {mi_mean:.3f}")
            print(f"Max MI: {mi_max:.3f}")
        
        # Generate visualizations sequentially for this category
        print(f"\nGenerating visualizations for {name}...")
        
        # 1. Entropy per Vertex visualization
        print("  -> Creating Entropy per Vertex visualization...")
        entropy_fig = create_vertex_entropy_visualization(mi_matrix, graph, name)
        if entropy_fig:
            plt.show()
            plt.close(entropy_fig)
        
        # 2. MI Matrix visualization
        print("  -> Creating Mutual Information Matrix visualization...")
        mi_fig = create_mi_matrix_visualization(mi_matrix, name)
        if mi_fig:
            plt.show()
            plt.close(mi_fig)
        
        print(f"  -> Visualizations complete for {name}")
    
    print("\n" + "="*80)
    print("TAXONOMY ANALYSIS COMPLETE")
    print("="*80)
    print(f"Analysis completed in: {time.time() - start_time:.2f} seconds")
    print(f"Canonical archetypes analyzed: {len(canonical_graphs)}")
    print("\nArchetype Summary:")
    for name, data in canonical_graphs.items():
        if 'graph' in data:
            graph = data['graph']
            print(f"  â¢ {name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
if __name__ == "__main__":
    run_taxonomy_analysis()
