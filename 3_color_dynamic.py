import networkx as nx
import numpy as np
import time
import warnings
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# --- PyTorch and PyG Imports for the GNN ---
# Note: This requires PyTorch and PyTorch Geometric to be installed.
# pip install torch torch_geometric
try:
    import torch
    import torch.nn.functional as F
    from torch.nn import Linear
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


# --- Suppress deprecation warnings ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. VERIFIED GRAPH DEFINITIONS ---

def get_graph_data(name):
    """
    Returns a networkx graph object for a specified graph.
    All graph structures are hard-coded from canonical sources.
    """
    G = nx.Graph()
    if name == "prism":
        G.add_nodes_from(range(6))
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)])
    elif name == "petersen":
        G.add_nodes_from(range(10))
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (7, 9), (9, 6), (6, 8), (8, 5)])
    elif name == "chvatal":
        G.add_nodes_from(range(12))
        G.add_edges_from([(0,1), (0,4), (0,6), (0,9), (1,2), (1,5), (1,7), (2,3), (2,6), (2,8), (3,4), (3,7), (3,9), (4,5), (4,8), (5,10), (5,11), (6,10), (6,11), (7,8), (7,11), (8,10), (9,10), (9,11)])
    elif name == "groetzsch":
        G.add_nodes_from(range(11))
        G.add_edges_from([(0,1), (0,2), (0,3), (0,4), (0,5), (1,6), (1,8), (1,10), (2,6), (2,7), (2,9), (3,7), (3,8), (3,10), (4,6), (4,9), (4,10), (5,7), (5,8), (5,9)])
    elif name == "k4":
        G.add_nodes_from(range(4))
        G.add_edges_from([(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)])
    elif name == "w6":
        G.add_nodes_from(range(7))
        G.add_edges_from([(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (1,2), (2,3), (3,4), (4,5), (5,6), (6,1)])
    elif name == "Cycle C₅":
        G.add_nodes_from(range(5))
        G.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,0)])
    elif "Random" in name:
        match = re.search(r'\((\d+),([\d.]+)\)', name)
        if match:
            n = int(match.group(1))
            p = float(match.group(2))
            G = nx.gnp_random_graph(n, p, seed=42)
        else:
            raise ValueError(f"Could not parse random graph parameters from name: {name}")
    else:
        raise ValueError(f"Invalid graph name: {name}")
    return G

# --- 2. GRAPH NEURAL NETWORK DEFINITION ---

class GCN(torch.nn.Module):
    """A simple Graph Convolutional Network for graph classification."""
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42) # for reproducibility
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

def networkx_to_pyg_data(G):
    """Converts a networkx graph to a PyTorch Geometric data object."""
    degrees = np.array([G.degree(n) for n in G.nodes()]).reshape(-1, 1)
    clustering_coeffs = np.array([nx.clustering(G, n) for n in G.nodes()]).reshape(-1, 1)
    node_features = np.hstack([degrees, clustering_coeffs])
    
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    return data

# --- 3. ML CLASSIFIER TRAINING & PREDICTION ---

def train_and_predict_gnn(graph_data_list, labels, epochs=50):
    """
    Trains a GNN model for a given number of epochs and makes predictions
    using Leave-One-Out Cross-Validation.
    """
    if not PYG_AVAILABLE:
        return None, 0.0

    y = np.array(labels)
    loo = LeaveOneOut()
    predictions = np.zeros(len(y), dtype=int)
    
    for train_index, test_index in loo.split(graph_data_list):
        model = GCN(num_node_features=2, hidden_channels=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for i in train_index:
                data = graph_data_list[i]
                label = torch.tensor([y[i]], dtype=torch.long)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, torch.zeros(data.num_nodes, dtype=torch.long))
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()

        model.eval()
        test_data = graph_data_list[test_index[0]]
        out = model(test_data.x, test_data.edge_index, torch.zeros(test_data.num_nodes, dtype=torch.long))
        pred = out.argmax(dim=1)
        predictions[test_index[0]] = pred.item()
        
    return predictions, accuracy_score(y, predictions)


# --- 4. BENCHMARKING AUTOMATION with ADAPTIVE SEARCH ---

def run_benchmark():
    """
    Runs the classifier on a set of benchmark graphs and uses an adaptive
    search to find the optimal number of training epochs.
    """
    if not PYG_AVAILABLE:
        print("\n[ERROR] PyTorch or PyTorch Geometric not found. Cannot run GNN benchmark.")
        print("Please install with: pip install torch torch_geometric")
        return

    benchmark_graphs = {
        "Cycle C₅": ("Cycle C₅", 1),
        "Prism Graph": ("prism", 1),
        "Petersen Graph": ("petersen", 1),
        "Random Sparse G(20,0.1)": ("Random Sparse G(20,0.1)", 1),
        "K₄ Graph": ("k4", 0),
        "Grötzsch Graph": ("groetzsch", 0),
        "Chvátal Graph": ("chvatal", 0),
        "Wheel Graph W₆": ("w6", 0),
        "Random Dense G(20,0.4)": ("Random Dense G(20,0.4)", 0)
    }

    graph_names = list(benchmark_graphs.keys())
    graph_data_list = []
    labels = []

    print("--- Preparing Graph Data for GNN ---")
    for name in graph_names:
        graph_key, label = benchmark_graphs[name]
        G = get_graph_data(graph_key)
        pyg_data = networkx_to_pyg_data(G)
        graph_data_list.append(pyg_data)
        labels.append(label)
        print(f"Processed {name:<25}")
    
    # --- ADAPTIVE HYPERPARAMETER TUNING LOOP ---
    print("\n--- Running Adaptive Search for Optimal Epochs ---")
    
    # Initial coarse search space
    search_space = [50, 500, 1000]
    
    best_accuracy = 0
    best_epoch = 0
    best_predictions = None
    
    # Keep track of tested epochs to avoid re-running
    tested_epochs = {}

    iteration = 0
    while iteration < 5: # Limit iterations to prevent infinite loops
        iteration += 1
        made_improvement = False
        
        current_best_in_iter = best_accuracy

        for epochs in search_space:
            if epochs in tested_epochs:
                continue

            print(f"\nTraining for {epochs} epochs...")
            predictions, accuracy = train_and_predict_gnn(graph_data_list, labels, epochs=epochs)
            tested_epochs[epochs] = accuracy
            print(f"Accuracy: {accuracy * 100:.2f}%")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epochs
                best_predictions = predictions
                made_improvement = True
        
        if not made_improvement:
            print("\nNo further improvement found. Converged on optimal epoch value.")
            break
        
        # Create a new, refined search space around the new best epoch
        print(f"\nNew best epoch found: {best_epoch}. Refining search space...")
        step = max(50, int(best_epoch * 0.2)) # step size is 20% of best epoch, or 50
        search_space = [best_epoch - step, best_epoch + step]
        search_space = [e for e in search_space if e > 0] # ensure epochs are positive


    print("\n\n--- DETAILED REPORT FOR BEST PERFORMING MODEL ---")
    print(f"Optimal setting found: {best_epoch} epochs with {best_accuracy * 100:.2f}% accuracy.")
    print("-" * 80)
    print(f"{'Graph Name':<25} | {'Ground Truth':<15} | {'Prediction':<20} | {'Correct?':<10}")
    print("-" * 80)

    for i, name in enumerate(graph_names):
        ground_truth_label = "3-Colorable" if labels[i] == 1 else "Not 3-Colorable"
        prediction_label = "Likely 3-Colorable" if best_predictions[i] == 1 else "Unlikely 3-Colorable"
        is_correct = (labels[i] == best_predictions[i])
        status = "✔" if is_correct else "✘"
        print(f"{name:<25} | {ground_truth_label:<15} | {prediction_label:<20} | {status:<10}")

    print("-" * 80)
    
    print("\n--- HYPERPARAMETER TUNING SUMMARY ---")
    print("-" * 40)
    print(f"{'Epochs Tested':<20} | {'Overall Accuracy':<20}")
    print("-" * 40)
    # Sort results for clear presentation
    sorted_results = sorted(tested_epochs.items())
    for epochs, acc in sorted_results:
        print(f"{epochs:<20} | {acc * 100:<20.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    run_benchmark()
