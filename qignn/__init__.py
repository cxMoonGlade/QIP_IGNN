"""
QIGNN - Quantum-Conditioned Implicit Graph Neural Network

Modules:
- ansatz: Quantum gates, state operations, Deep XYZ circuit, qubit topologies
- quantum_torch: TorchQuantumLayer, TopoAwareQuantumLayer
- lqa: Local Quantum Aggregator (entanglement-based GIN aggregation)
- topology: Cycle basis extraction, topological features
- model: GINEncoder, BatchedImplicitCore, TopoAwareQIGNN, pooling, decoder
"""

from .topology import (
    precompute_topology_features,
    TopologyFeatureExtractor,
)
from .ansatz import get_edge_pairs, TorchQuantumCircuit
from .quantum_torch import TorchQuantumLayer, TopoAwareQuantumLayer
from .lqa import (
    LocalQuantumAggregator,
    LocalQuantumGNN,
    LocalQuantumGINEncoder,
)
from .model import (
    GINEncoder,
    SimpleEncoder,
    MinEncoder,
    BatchedGraphPooling,
    BatchedImplicitCore,
    GNNDecoder,
    TopoAwareQIGNN,
)

__all__ = [
    'precompute_topology_features',
    'TopologyFeatureExtractor',
    'get_edge_pairs',
    'TorchQuantumCircuit',
    'TorchQuantumLayer',
    'TopoAwareQuantumLayer',
    'LocalQuantumAggregator',
    'LocalQuantumGNN',
    'LocalQuantumGINEncoder',
    'GINEncoder',
    'SimpleEncoder',
    'MinEncoder',
    'BatchedGraphPooling',
    'BatchedImplicitCore',
    'GNNDecoder',
    'TopoAwareQIGNN',
]
