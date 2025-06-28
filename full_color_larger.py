import networkx as nx
import itertools
from collections import defaultdict
import time
import concurrent.futures

# -- Configuration --
# The colors are represented by integers 0, 1, 2
COLORS = [0, 1, 2] 
# The two "measurement settings" for each party will be colors 0 and 1
SETTINGS = [0, 1] 

# --- Graph Coloring Engine ---

def find_all_3_colorings_recursive(graph, nodes, current_coloring, all_colorings):
    """Recursively finds all valid 3-colorings for a graph."""
    if not nodes:
        all_colorings.append(dict(current_coloring))
        return

    node = nodes[0]
    remaining_nodes = nodes[1:]
    
    neighbor_colors = {current_coloring[neighbor] for neighbor in graph.neighbors(node) if neighbor in current_coloring}

    for color in COLORS:
        if color not in neighbor_colors:
            current_coloring[node] = color
            find_all_3_colorings_recursive(graph, remaining_nodes, current_coloring, all_colorings)
            del current_coloring[node] # Backtrack

# --- Bell Test Logic ---

def calculate_expectation_value(all_colorings, u, v, setting_u, setting_v):
    """
    Calculates the expectation value E(a,b) for a given pair of settings.
    Outcome is +1 if node's color matches the setting, -1 otherwise.
    """
    if not all_colorings:
        return 0.0

    total_product = sum(
        (1 if c.get(u) == setting_u else -1) * (1 if c.get(v) == setting_v else -1)
        for c in all_colorings
    )
    
    return total_product / len(all_colorings)

def select_distant_nodes(graph):
    """Selects a starting node and a node maximally distant from it."""
    if not list(graph.nodes()):
        return None, None, 0
    
    nodes = list(graph.nodes())
    if not nx.is_connected(graph):
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc).copy()
        if not list(subgraph.nodes()): return nodes[0], nodes[0], 0
        
        node_u = list(subgraph.nodes())[0]
        lengths = nx.shortest_path_length(subgraph, source=node_u)
        node_v = max(lengths, key=lengths.get)
        return node_u, node_v, lengths[node_v]

    node_u = nodes[0]
    lengths = nx.shortest_path_length(graph, source=node_u)
    node_v = max(lengths, key=lengths.get)
    return node_u, node_v, lengths[node_v]

def run_bell_test_for_graph(graph_name, graph_generator):
    """Runs the full Bell test analysis for a single graph."""
    
    graph = graph_generator()
    print(f"--- Starting 3-State Graph Bell Test for: {graph_name} ---")
    start_time = time.time()
    
    all_colorings = []
    find_all_3_colorings_recursive(graph, list(graph.nodes()), {}, all_colorings)
    num_colorings = len(all_colorings)
    print(f"> Found {num_colorings} valid 3-colorings.")

    if num_colorings == 0:
        return {
            "name": graph_name, "v": graph.number_of_nodes(), "e": graph.number_of_edges(),
            "bell_value": 0.0, "verdict": "✘ Respected", "conclusion": "Not 3-colorable"
        }

    u, v, dist = select_distant_nodes(graph)
    
    setting_a, setting_a_prime = SETTINGS[0], SETTINGS[1]
    setting_b, setting_b_prime = SETTINGS[0], SETTINGS[1]

    E_ab = calculate_expectation_value(all_colorings, u, v, setting_a, setting_b)
    E_ab_prime = calculate_expectation_value(all_colorings, u, v, setting_a, setting_b_prime)
    E_a_prime_b = calculate_expectation_value(all_colorings, u, v, setting_a_prime, setting_b)
    E_a_prime_b_prime = calculate_expectation_value(all_colorings, u, v, setting_a_prime, setting_b_prime)

    bell_value = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime
    
    end_time = time.time()
    print(f"Finished calculation in {end_time - start_time:.2f} seconds.")
    
    if abs(bell_value) <= 2.0001:
        verdict = "✘ Respected"
        conclusion = "Consistent with local model"
    else:
        verdict = "✔ VIOLATED!"
        conclusion = "Correlations EXCEED classical limit!"
    
    return {
        "name": graph_name, "v": graph.number_of_nodes(), "e": graph.number_of_edges(),
        "bell_value": bell_value, "verdict": verdict, "conclusion": conclusion
    }

# --- Top-Level Graph Generator Functions (for pickling) ---

def create_cycle_c5(): return nx.cycle_graph(5)
def create_prism_graph():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)])
    return g
def create_petersen_graph(): return nx.petersen_graph()
def create_k4_graph(): return nx.complete_graph(4)
def create_groetzsch_graph(): return nx.groetzsch_graph()
def create_chvatal_graph(): return nx.chvatal_graph()
def create_k10_graph(): return nx.complete_graph(10)
def create_turan_graph(): return nx.turan_graph(15, 4)
def create_mycielski_graph(): return nx.mycielski_graph(5)
def create_kneser_graph(): return nx.kneser_graph(7, 3)
def create_grid_graph(): return nx.grid_2d_graph(10, 10)
def create_barbell_graph(): return nx.barbell_graph(7, 0)
    
# --- Main Execution ---

if __name__ == "__main__":
    graph_generators = {
        "Cycle C₅": create_cycle_c5,
        "Prism": create_prism_graph,
        "Petersen": create_petersen_graph,
        "K₄": create_k4_graph,
        "Grötzsch": create_groetzsch_graph,
        "Chvátal": create_chvatal_graph,
        "Complete K₁₀": create_k10_graph,
        "Turán T(15, 4)": create_turan_graph,
        "Mycielski M₅": create_mycielski_graph,
        "Kneser KG(7, 3)": create_kneser_graph,
        "Grid 10x10": create_grid_graph,
        "Barbell (2x K₇)": create_barbell_graph
    }
    
    all_results = []
    
    # Using a ProcessPoolExecutor to run tests in parallel.
    # The 'if __name__ == "__main__"' block is essential for this to work correctly on all platforms.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_graph = {executor.submit(run_bell_test_for_graph, name, gen): name for name, gen in graph_generators.items()}
        for future in concurrent.futures.as_completed(future_to_graph):
            graph_name = future_to_graph[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                    print(f"✓ Completed test for: {graph_name}")
            except Exception as exc:
                print(f"✗ {graph_name} generated an exception: {exc}")

    all_results.sort(key=lambda x: x['name'])

    print("\n\n--- Aggregate Results Table ---")
    print(f"{'Graph Name':<20} | {'Vertices (V)':>12} | {'Edges (E)':>10} | {'Final Bell Value':>18} | {'Verdict':>12} | {'Conclusion'}")
    print(f"{'-'*20}-+-{'-'*12}-+-{'-'*10}-+-{'-'*18}-+-{'-'*12}-+-{'-'*30}")
    for res in all_results:
        print(f"{res['name']:<20} | {res['v']:>12} | {res['e']:>10} | {res['bell_value']:>18.4f} | {res['verdict']:>12} | {res['conclusion']}")

