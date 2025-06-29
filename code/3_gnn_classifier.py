# FILE: 3_gnn_classifier.py
# DESC: Implements a Graph Neural Network (GNN) classifier for 3-colorability.
#       This module corresponds to the conclusion of the paper (Section 5), which
#       argues that the existence of "constraint entanglement" justifies the
#       necessity of learning-based models like GNNs that can capture the
#       complex, non-local correlations inherent to the problem.

import networkx as nx
import numpy as np
import warnings

# This script requires PyTorch and PyTorch Geometric.
# pip install torch torch_geometric
try:
    import torch
    import torch.nn.functional as F
    from torch.nn import Linear
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, DataLoader
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Graph Definitions & Feature Extraction ---
def get_benchmark_graph(name, label):
    """ Returns a networkx graph and its ground-truth 3-colorability label. """
    G = nx.Graph()
    # 3-Colorable Graphs (Label = 1)
    if name == "prism": G.add_edges_from([(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)])
    elif name == "petersen": G = nx.petersen_graph()
    elif name == "Cycle C5": G = nx.cycle_graph(5)
    # Not 3-Colorable Graphs (Label = 0)
    elif name == "chvatal": G = nx.chvatal_graph()
    elif name == "groetzsch": G.add_nodes_from(range(11)); G.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(1,6),(1,8),(1,10),(2,6),(2,7),(2,9),(3,7),(3,8),(3,10),(4,6),(4,9),(4,10),(5,7),(5,8),(5,9)])
    elif name == "k4": G = nx.complete_graph(4)
    else: raise ValueError(f"Invalid graph name: {name}")
    return G, label

def networkx_to_pyg_data(G, y):
    """
    Converts a networkx graph to a PyTorch Geometric data object.
    The node features (degree, clustering coeff) are simple, local "hidden variables"
    of the type discussed in Section 2.2. The GNN's task is to learn a better
    representation from this basic information.
    """
    # Simple local features
    degrees = np.array([G.degree(n) for n in G.nodes()]).reshape(-1, 1)
    clustering = np.array([nx.clustering(G, n) for n in G.nodes()]).reshape(-1, 1)
    node_features = np.hstack([degrees, clustering])
    
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    label = torch.tensor([y], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=label)

# --- Graph Neural Network Definition ---
class GCN(torch.nn.Module):
    """
    A simple Graph Convolutional Network for graph classification.
    Its message-passing layers are designed to learn the complex, non-local
    correlations that the paper argues are characteristic of NP-complete problems.
    """
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels * 2)
        self.lin = Linear(hidden_channels * 2, 2) # Output classes: 0 or 1

    def forward(self, x, edge_index, batch):
        # Message Passing Layers
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        # Readout Layer
        x = global_mean_pool(x, batch)
        # Final Classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

def run_gnn_benchmark():
    if not PYG_AVAILABLE:
        print("[ERROR] PyTorch or PyTorch Geometric not found.")
        print("Please run: pip install torch torch_geometric")
        return

    print("\n--- GNN Classifier Benchmark ---")
    print("Justification for a learning-based approach (Conclusion, Section 5)\n")
    
    benchmark_graphs = [
        ("Cycle C5", 1), ("Prism", 1), ("Petersen Graph", 1),
        ("K4", 0), ("Grötzsch", 0), ("Chvátal", 0)
    ]
    
    dataset = [networkx_to_pyg_data(*get_benchmark_graph(name, label)) for name, label in benchmark_graphs]
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = GCN(num_node_features=2, hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    print("Training GNN to learn non-local features...")
    for epoch in range(100):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
    print("Training complete.\n")

    print("--- GNN Performance on Benchmark Graphs ---")
    print(f"{'Graph Name':<20} | {'Ground Truth':<15} | {'GNN Prediction':<20} | Status")
    print("-" * 80)
    
    model.eval()
    for i, data in enumerate(test_loader):
        graph_name = benchmark_graphs[i][0]
        true_label = "3-Colorable" if data.y.item() == 1 else "Not 3-Colorable"
        
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1).item()
        
        pred_label = "3-Colorable" if pred == 1 else "Not 3-Colorable"
        status = "✔ Correct" if pred == data.y.item() else "✘ Incorrect"
        print(f"{graph_name:<20} | {true_label:<15} | {pred_label:<20} | {status}")
    print("-" * 80)


if __name__ == "__main__":
    run_gnn_benchmark()
