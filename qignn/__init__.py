"""
*Quantum injection pathways* for implicit GNNs (see package :mod:`qignn.model`).

Subpackages: ``ansatz`` (Deep XYZ circuit), ``quantum_torch`` (encode--unitary--measure
maps), ``topology`` (cycle features :math:`\\tau`), ``lqa`` (optional LQA encoder),
and ``model`` (encoders, :class:`BatchedImplicitCore`, :class:`TopoAwareQIGNN`).
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
