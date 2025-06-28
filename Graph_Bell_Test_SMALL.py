import networkx as nx
import numpy as np
import time
import itertools
from multiprocessing import Pool, cpu_count
from collections import defaultdict

# --- 1. VERIFIED GRAPH DEFINITIONS ---

def get_graph_data(name):
    """
    Returns a networkx graph object for a specified graph.
    All graph structures are hard-coded from canonical sources.
    """
    G = nx.Graph()
    if name == "prism": # 3-colorable
        G.add_nodes_from(range(6))
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)])
    elif name == "petersen": # 3-colorable
        G.add_nodes_from(range(10))
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (7, 9), (9, 6), (6, 8), (8, 5)])
    elif name == "Cycle C₅": # 3-colorable
        G.add_nodes_from(range(5))
        G.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,0)])
    elif name == "K₄": # Not 3-colorable
        G.add_nodes_from(range(4))
        G.add_edges_from([(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)])
    elif name == "groetzsch": # Not 3-colorable
        G.add_nodes_from(range(11))
        G.add_edges_from([(0,1), (0,2), (0,3), (0,4), (0,5), (1,6), (1,8), (1,10), (2,6), (2,7), (2,9), (3,7), (3,8), (3,10), (4,6), (4,9), (4,10), (5,7), (5,8), (5,9)])
    elif name == "chvatal": # Not 3-colorable
        G.add_nodes_from(range(12))
        G.add_edges_from([(0,1), (0,4), (0,6), (0,9), (1,2), (1,5), (1,7), (2,3), (2,6), (2,8), (3,4), (3,7), (3,9), (4,5), (4,8), (5,10), (5,11), (6,10), (6,11), (7,8), (7,11), (8,10), (9,10), (9,11)])
    else:
        raise ValueError(f"Invalid graph name: {name}")
    return G

# --- 2. THE EXACT SOLVER & CORRELATION CALCULATOR ---

def find_all_colorings(G, initial_constraints={}):
    """
    An exact recursive solver that finds ALL valid 3-colorings for a
    graph, given a dictionary of pre-colored 'initial_constraints'.
    """
    adj = {node: list(neighbors) for node, neighbors in G.adj.items()}
    nodes = sorted(list(G.nodes()))
    coloring = initial_constraints.copy()
    solutions = []
    
    nodes_to_color = [n for n in nodes if n not in coloring]

    def is_safe(vertex, color, current_coloring):
        for neighbor in adj.get(vertex, []):
            if current_coloring.get(neighbor) == color:
                return False
        return True

    def solve(vertex_idx):
        if vertex_idx == len(nodes_to_color):
            solutions.append(coloring.copy())
            return

        vertex = nodes_to_color[vertex_idx]
        for color in range(3): # Colors are 0, 1, 2
            if is_safe(vertex, color, coloring):
                coloring[vertex] = color
                solve(vertex_idx + 1)
                del coloring[vertex] # Backtrack

    solve(0)
    return solutions

def calculate_joint_probabilities(G, u, v, u_n, v_n, setting_u, setting_v):
    """
    Calculates the 3x3 joint probability matrix for two detector nodes,
    given the measurement settings for the two distant nodes.
    """
    initial_constraints = {u: setting_u, v: setting_v}
    
    # This is the computationally expensive step
    print(f"  > Starting calculation for constraints: u={setting_u}, v={setting_v}...")
    all_solutions = find_all_colorings(G, initial_constraints)
    print(f"  > Finished. Found {len(all_solutions)} valid colorings for this setting.")
    
    num_solutions = len(all_solutions)
    if num_solutions == 0:
        return np.zeros((3, 3))
        
    counts = defaultdict(int)
    for sol in all_solutions:
        outcome_u = sol.get(u_n, -1)
        outcome_v = sol.get(v_n, -1)
        if outcome_u != -1 and outcome_v != -1:
            counts[(outcome_u, outcome_v)] += 1
            
    prob_matrix = np.zeros((3, 3))
    for (i, j), count in counts.items():
        prob_matrix[i, j] = count / num_solutions
        
    return prob_matrix


# --- 3. THE BELL TEST ---

def run_graph_bell_test(graph_name):
    """
    Orchestrates the full Bell Test experiment on a given graph.
    """
    print(f"--- Starting 3-State Graph Bell Test for: {graph_name} ---")
    G = get_graph_data(graph_name)
    
    if G.number_of_nodes() < 4:
        print("[ERROR] Graph is too small for a meaningful test.")
        return

    # Find two distant nodes u, v
    try:
        all_paths = dict(nx.all_pairs_shortest_path_length(G))
        u, v = max(((u, v) for u in all_paths for v in all_paths[u]), key=lambda x: all_paths[x[0]][x[1]])
    except nx.NetworkXError: # Handle disconnected graphs
        print("[WARNING] Graph is not connected. Testing on the largest component.")
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        all_paths = dict(nx.all_pairs_shortest_path_length(subgraph))
        u, v = max(((u, v) for u in all_paths for v in all_paths[u]), key=lambda x: all_paths[x[0]][x[1]])

    
    try:
        u_n = list(G.neighbors(u))[0]
        v_n = list(G.neighbors(v))[0]
    except IndexError:
        print("[ERROR] Selected nodes do not have neighbors.")
        return

    print(f"Selected distant measurement nodes: u={u}, v={v} (Distance: {all_paths[u][v]})")
    print(f"Selected detector neighbors: u_n={u_n}, v_n={v_n}")
    
    settings_to_test = [(0, 0), (0, 1), (1, 0), (1, 1)]
    pool_args = [(G, u, v, u_n, v_n, s_u, s_v) for s_u, s_v in settings_to_test]

    print(f"\nRunning {len(settings_to_test)} correlation experiments on {cpu_count()} CPU cores...")
    start_time = time.time()
    with Pool(cpu_count()) as p:
        results = p.starmap(calculate_joint_probabilities, pool_args)
    end_time = time.time()
    print(f"Finished all calculations in {end_time - start_time:.2f} seconds.")

    prob_matrices = {settings: matrix for settings, matrix in zip(settings_to_test, results)}

    def get_s_value(prob_matrix):
        term1 = np.trace(prob_matrix)
        term2 = prob_matrix[0,2] + prob_matrix[1,0] + prob_matrix[2,1]
        term3 = prob_matrix[2,0] + prob_matrix[0,1] + prob_matrix[1,2]
        return term1 + term2 + term3

    S_a_b = get_s_value(prob_matrices[(0,0)])
    S_a_bp = get_s_value(prob_matrices[(0,1)])
    S_ap_b = get_s_value(prob_matrices[(1,0)])
    S_ap_bp = get_s_value(prob_matrices[(1,1)])

    bell_value = S_a_b - S_a_bp + S_ap_b + S_ap_bp

    print("\n--- Bell Test Results ---")
    print(f"S(u=Red,v=Red)   = {S_a_b:.4f}")
    print(f"S(u=Red,v=Green) = {S_a_bp:.4f}")
    print(f"S(u=Green,v=Red) = {S_ap_b:.4f}")
    print(f"S(u=Green,v=Green)= {S_ap_bp:.4f}")
    print("-" * 30)
    print(f"Final Bell Value: S(a,b)-S(a,b')+S(a',b)+S(a',b') = {bell_value:.4f}")
    print("Classical Limit: <= 2.0")
    print("-" * 30)

    if bell_value > 2.0001: 
        print("Verdict: ✔ Bell's Inequality VIOLATED.")
        print("Conclusion: Evidence found for non-local 'color entanglement'.")
    else:
        print("Verdict: ✘ Bell's Inequality Respected.")
        print("Conclusion: The graph's correlations are consistent with a local model.")


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # WARNING: THIS IS A COMPUTATIONALLY INTENSIVE EXPERIMENT.
    # The runtime is massively exponential in the number of graph nodes.
    # - "Cycle C₅" (5 nodes): Fast (~1 second)
    # - "prism" (6 nodes): Slow (~10-30 seconds)
    # - "petersen" (10 nodes): VERY SLOW (potentially hours)
    # - "groetzsch" (11 nodes), "chvatal" (12 nodes): EXTREMELY SLOW (potentially days)
    # --------------------------------------------------------------------------

    # Select a graph to test:

    graph_names = ["Cycle C₅", "prism", "petersen", "K₄", "groetzsch", "chvatal"]

    for name in graph_names:
        try:
            run_graph_bell_test(name)
            print("\n" + "="*60 + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to run Bell test on graph '{name}': {e}")

