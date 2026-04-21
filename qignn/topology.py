r"""
Graph-side topology descriptors :math:`\tau(A)` for the paper (independent
injection :math:`Q_{\mathrm{IN}}(H,\tau(A))` and, when used, modulations on the
encode--unitary--measure map).  Builds a (minimum) cycle basis and per-node / graph
cycle-count features consumed by :class:`qignn.quantum_torch.TopoAwareQuantumLayer`.
The optional PCB-GNN-style gating in that layer follows the *PCB-GNN* reference
cited in the paper’s related work.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
import networkx as nx
from torch_geometric.data import Data, Batch


def compute_cycle_basis(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Compute minimum cycle basis of a graph via NetworkX."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.cpu().numpy().T
    G.add_edges_from(edges)
    try:
        cycles = nx.cycle_basis(G)
    except Exception:
        cycles = []
    return cycles


def extract_node_cycle_features(
    cycles: List[List[int]], num_nodes: int, max_cycle_length: int = 20
) -> torch.Tensor:
    """
    For each node v, compute f_l(v) = number of basis cycles of length l containing v.

    Returns: [num_nodes, max_cycle_length - 2] (lengths 3..max_cycle_length)
    """
    feature_dim = max_cycle_length - 2
    features = torch.zeros(num_nodes, feature_dim)

    for cycle in cycles:
        cycle_len = len(cycle)
        if cycle_len > max_cycle_length:
            continue
        len_idx = cycle_len - 3
        if 0 <= len_idx < feature_dim:
            for node in cycle:
                if node < num_nodes:
                    features[node, len_idx] += 1

    return features


def extract_graph_cycle_features(
    cycles: List[List[int]], max_cycle_length: int = 20
) -> torch.Tensor:
    """
    Graph-level cycle statistics: [n_cycles, min_len, max_len, mean_len] + length histogram.

    Returns: [4 + max_cycle_length - 2]
    """
    feature_dim = max_cycle_length - 2
    histogram = torch.zeros(feature_dim)

    if len(cycles) == 0:
        return torch.cat([torch.tensor([0.0, 0.0, 0.0, 0.0]), histogram])

    cycle_lengths = [len(c) for c in cycles]
    for length in cycle_lengths:
        len_idx = length - 3
        if 0 <= len_idx < feature_dim:
            histogram[len_idx] += 1
    histogram = histogram / (len(cycles) + 1e-6)

    stats = torch.tensor([
        len(cycles), min(cycle_lengths), max(cycle_lengths),
        np.mean(cycle_lengths)
    ], dtype=torch.float32)

    return torch.cat([stats, histogram])


def extract_node_structural_features(
    edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """
    Per-node structural features: [degree, clustering_coeff, in_triangle].

    Returns: [num_nodes, 3]
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.cpu().numpy().T
    G.add_edges_from(edges)

    features = torch.zeros(num_nodes, 3)
    max_degree = max(num_nodes - 1, 1)

    for node in range(num_nodes):
        features[node, 0] = G.degree(node) / max_degree

    clustering = nx.clustering(G)
    for node in range(num_nodes):
        features[node, 1] = clustering.get(node, 0)

    triangles = nx.triangles(G)
    for node in range(num_nodes):
        features[node, 2] = 1.0 if triangles.get(node, 0) > 0 else 0.0

    return features


class TopologyFeatureExtractor(nn.Module):
    """
    Complete topology feature extractor with caching.

    Extracts node-level cycle features, graph-level cycle features,
    and optional structural features for each graph.
    """

    def __init__(
        self,
        max_cycle_length: int = 20,
        use_structural: bool = True,
        cache_features: bool = True,
    ):
        super().__init__()

        self.max_cycle_length = max_cycle_length
        self.use_structural = use_structural
        self.cache_features = cache_features

        self.node_cycle_dim = max_cycle_length - 2
        self.graph_cycle_dim = 4 + self.node_cycle_dim
        self.struct_dim = 3 if use_structural else 0
        self.node_feat_dim = self.node_cycle_dim + self.struct_dim

        self._cache: Dict[int, Dict] = {}

    def forward(self, data: Data, graph_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if self.cache_features and graph_id is not None and graph_id in self._cache:
            return self._cache[graph_id]

        edge_index = data.edge_index
        num_nodes = data.num_nodes

        cycles = compute_cycle_basis(edge_index, num_nodes)
        node_cycle_feat = extract_node_cycle_features(cycles, num_nodes, self.max_cycle_length)
        graph_cycle_feat = extract_graph_cycle_features(cycles, self.max_cycle_length)

        result = {
            'node_cycle_features': node_cycle_feat,
            'graph_cycle_features': graph_cycle_feat,
            'num_cycles': len(cycles),
            'cycles': cycles,
        }

        if self.use_structural:
            struct_feat = extract_node_structural_features(edge_index, num_nodes)
            result['node_structural_features'] = struct_feat
            result['combined_node_features'] = torch.cat([node_cycle_feat, struct_feat], dim=-1)
        else:
            result['combined_node_features'] = node_cycle_feat

        if self.cache_features and graph_id is not None:
            self._cache[graph_id] = result

        return result

    def clear_cache(self):
        self._cache.clear()


def precompute_topology_features(
    dataset,
    max_cycle_length: int = 20,
    use_structural: bool = True,
    verbose: bool = True,
    return_stats: bool = False,
) -> 'List[Dict[str, torch.Tensor]]':
    """
    Precompute topology features for an entire dataset (one-time O(n^3) per graph).

    If return_stats=True, returns (features_list, stats_dict) with degree statistics.
    """
    extractor = TopologyFeatureExtractor(
        max_cycle_length=max_cycle_length,
        use_structural=use_structural,
        cache_features=False,
    )

    all_features = []
    n_total = len(dataset)
    milestones = {int(n_total * p): f"{int(p*100)}%" for p in [0.25, 0.5, 0.75, 1.0]}

    max_degrees = []
    avg_degrees = []

    for i, data in enumerate(dataset):
        features = extractor(data, graph_id=i)
        all_features.append(features)

        edge_index = data.edge_index
        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else edge_index.max().item() + 1
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        degrees.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
        max_degrees.append(degrees.max().item())
        avg_degrees.append(degrees.float().mean().item())

        if verbose and (i + 1) in milestones:
            print(f"  {milestones[i + 1]} ({i + 1}/{n_total})")

    n_cycles_list = [f['num_cycles'] for f in all_features]
    global_max_degree = max(max_degrees)
    global_avg_degree = np.mean(avg_degrees)
    p95_degree = int(np.percentile(max_degrees, 95))

    if verbose:
        print(f"\nTopology Statistics:")
        print(f"  Graphs with cycles: {sum(1 for n in n_cycles_list if n > 0)}/{len(dataset)}")
        print(f"  Avg cycles per graph: {np.mean(n_cycles_list):.2f}")
        print(f"  Max cycles: {max(n_cycles_list)}")
        print(f"  Max degree (neighbors): {global_max_degree}")
        print(f"  95th percentile degree: {p95_degree}")
        print(f"  Avg degree: {global_avg_degree:.2f}")
        print(f"  Node feature dim: {extractor.node_feat_dim}")
        print(f"  Graph feature dim: {extractor.graph_cycle_dim}")

    if return_stats:
        stats = {
            'max_degree': global_max_degree,
            'avg_degree': global_avg_degree,
            'p95_degree': p95_degree,
        }
        return all_features, stats

    return all_features
