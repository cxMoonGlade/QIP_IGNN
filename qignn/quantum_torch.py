"""
Quantum Neural Network Layers.

Wraps the quantum circuit ansatz into nn.Module layers:
- TorchQuantumLayer: drop-in quantum feature layer
- TopoAwareQuantumLayer: topology-conditioned quantum layer (PCB-GNN inspired)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from .ansatz import (
    rx_matrix, ry_matrix, rz_matrix,
    ising_zz_matrix, ising_xx_matrix, ising_yy_matrix,
    apply_single_qubit_gate, apply_two_qubit_gate,
    TorchQuantumCircuit,
)


class TorchQuantumLayer(nn.Module):
    """
    Drop-in quantum feature layer.

    Projects hidden features to qubit space, executes the Deep XYZ circuit,
    and projects back to hidden space. Matches the PennyLane QuantumLayer API.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_qubits: int,
        edge_pairs: List[Tuple[int, int]],
        circuit_reps: int = 1,
        graph_conditioned: bool = False,
        perm_invariant: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.graph_conditioned = graph_conditioned

        self.q_proj = nn.Linear(hidden_dim, n_qubits)
        self.q_ln = nn.LayerNorm(n_qubits) if use_layer_norm else nn.Identity()

        self.circuit = TorchQuantumCircuit(
            n_qubits=n_qubits,
            edge_pairs=edge_pairs,
            circuit_reps=circuit_reps,
            graph_conditioned=graph_conditioned,
            hidden_dim=hidden_dim,
            perm_invariant=perm_invariant,
        )

        self.q_out = nn.Linear(n_qubits, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        graph_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_dim] or [N, hidden_dim]
            graph_embedding: optional [batch, hidden_dim]
        Returns:
            [batch, hidden_dim]
        """
        q_input = self.q_proj(x)
        q_input = self.q_ln(q_input)
        q_input = torch.tanh(q_input)
        q_output = self.circuit(q_input, graph_embedding)
        return self.q_out(q_output)


class TopoAwareQuantumLayer(nn.Module):
    """
    Topology-Aware Quantum Layer (PCB-GNN inspired).

    Combines:
    1. Cycle basis features -> Ising interaction modulation
    2. Node structural features -> Data encoding modulation
    3. Competitive gating between topology-conditioned and fixed parameters
    """

    def __init__(
        self,
        hidden_dim: int,
        n_qubits: int,
        edge_pairs: List[Tuple[int, int]],
        circuit_reps: int = 1,
        topo_node_dim: int = 21,
        topo_graph_dim: int = 22,
        use_topo_encoding: bool = True,
        use_topo_ising: bool = True,
        use_competitive_gate: bool = True,
        perm_invariant: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.n_edges = len(edge_pairs)
        self.circuit_reps = circuit_reps
        self.use_topo_encoding = use_topo_encoding
        self.use_topo_ising = use_topo_ising
        self.use_competitive_gate = use_competitive_gate
        self.perm_invariant = perm_invariant

        self.q_proj = nn.Linear(hidden_dim, n_qubits)
        self.q_ln = nn.LayerNorm(n_qubits) if use_layer_norm else nn.Identity()

        self.circuit = TorchQuantumCircuit(
            n_qubits=n_qubits,
            edge_pairs=edge_pairs,
            circuit_reps=circuit_reps,
            graph_conditioned=False,
            hidden_dim=hidden_dim,
            perm_invariant=perm_invariant,
        )

        if use_topo_ising:
            n_ising_params = circuit_reps * 3 * self.n_edges
            self.ising_net = nn.Sequential(
                nn.Linear(topo_graph_dim, 64),
                nn.GELU(),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, n_ising_params),
                nn.Tanh(),
            )

        if use_topo_encoding:
            n_encoding_params = n_qubits * 3
            self.encoding_net = nn.Sequential(
                nn.Linear(topo_node_dim, 64),
                nn.GELU(),
                nn.Linear(64, n_encoding_params),
                nn.Tanh(),
            )

        if use_competitive_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(topo_graph_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            self.fixed_branch = nn.Linear(n_qubits, n_qubits)

        self.q_out = nn.Linear(n_qubits, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        topo_node_features: Optional[torch.Tensor] = None,
        topo_graph_features: Optional[torch.Tensor] = None,
        enable_encoding: bool = True,
        enable_ising: bool = True,
        enable_gate: bool = True,
    ) -> torch.Tensor:
        """
        Forward with topology conditioning and stochastic dropout support.

        Args:
            x: [batch, hidden_dim]
            topo_node_features: [batch, n_qubits, topo_node_dim]
            topo_graph_features: [batch, topo_graph_dim]
            enable_encoding/ising/gate: stochastic conditioning toggles
        """
        batch_size = x.shape[0]

        q_input = self.q_proj(x)
        q_input = self.q_ln(q_input)
        q_input = torch.tanh(q_input)

        # 1. Encoding modulation
        if self.use_topo_encoding and topo_node_features is not None and enable_encoding:
            node_agg = topo_node_features.mean(dim=1)
            encoding_mod = self.encoding_net(node_agg).view(batch_size, self.n_qubits, 3)
            scale_x = 1 + encoding_mod[:, :, 0] * 0.5
            scale_y = 1 + encoding_mod[:, :, 1] * 0.5
            scale_z = 1 + encoding_mod[:, :, 2] * 0.5
            avg_scale = (scale_x + scale_y + scale_z) / 3
            q_input = q_input * avg_scale

        # 2. Ising modulation
        ising_mod = None
        if self.use_topo_ising and topo_graph_features is not None and enable_ising:
            ising_params = self.ising_net(topo_graph_features)
            ising_mod = ising_params.view(batch_size, self.circuit_reps, 3, self.n_edges)

        # 3. Circuit execution
        if ising_mod is not None:
            q_output = self._circuit_with_ising_mod(q_input, ising_mod)
        else:
            q_output = self.circuit(q_input, graph_embedding=None)

        # 4. Competitive gating
        if self.use_competitive_gate and topo_graph_features is not None and enable_gate:
            gate = self.gate_net(topo_graph_features)
            fixed_out = self.fixed_branch(torch.tanh(self.q_proj(x)))
            q_output = gate * q_output + (1 - gate) * fixed_out

        return self.q_out(q_output)

    def _circuit_with_ising_mod(
        self, data: torch.Tensor, ising_mod: torch.Tensor
    ) -> torch.Tensor:
        """Execute circuit with external Ising weight modulation."""
        batch_size = data.shape[0]
        device = data.device
        circuit = self.circuit

        state = circuit._init_state(batch_size, device)
        rx_gates, ry_gates, rz_gates = circuit._precompute_encoding_gates(data)

        pi = circuit.perm_invariant
        for r in range(circuit.R):
            # Block 1: ZZ
            state = circuit._apply_data_encoding_cached(
                state, rx_gates, ry_gates, rz_gates, block_type='full')
            for edge_idx, (i, j) in enumerate(circuit.edge_pairs):
                idx = 0 if pi else edge_idx
                base = circuit.edge_weights_zz[r, idx]
                mod = ising_mod[:, r, 0, edge_idx]
                angle = circuit.eta[r, 0] * base * (1 + mod)
                state = apply_two_qubit_gate(
                    state, ising_zz_matrix(angle), i, j, circuit.n_qubits)
            nb_z = circuit.node_biases_z[r, :].expand(circuit.n_qubits)
            angles_z = circuit.eta[r, 0] * nb_z.unsqueeze(0).expand(batch_size, -1)
            rz_bias = rz_matrix(angles_z)
            for v in range(circuit.n_qubits):
                state = apply_single_qubit_gate(state, rz_bias[:, v], v, circuit.n_qubits)
            state = circuit._apply_trainable_rotations(state, r, 0)
            state = circuit._apply_cnot_ladder(state)

            # Block 2: XX
            state = circuit._apply_data_encoding_cached(
                state, rx_gates, ry_gates, rz_gates, block_type='xy')
            for edge_idx, (i, j) in enumerate(circuit.edge_pairs):
                idx = 0 if pi else edge_idx
                base = circuit.edge_weights_xx[r, idx]
                mod = ising_mod[:, r, 1, edge_idx]
                angle = circuit.eta[r, 1] * base * (1 + mod)
                state = apply_two_qubit_gate(
                    state, ising_xx_matrix(angle), i, j, circuit.n_qubits)
            nb_x = circuit.node_biases_x[r, :].expand(circuit.n_qubits)
            angles_x = circuit.eta[r, 1] * nb_x.unsqueeze(0).expand(batch_size, -1)
            rx_bias = rx_matrix(angles_x)
            for v in range(circuit.n_qubits):
                state = apply_single_qubit_gate(state, rx_bias[:, v], v, circuit.n_qubits)
            state = circuit._apply_trainable_rotations(state, r, 1)
            state = circuit._apply_cnot_ladder(state)

            # Block 3: YY
            state = circuit._apply_data_encoding_cached(
                state, rx_gates, ry_gates, rz_gates, block_type='yz')
            for edge_idx, (i, j) in enumerate(circuit.edge_pairs):
                idx = 0 if pi else edge_idx
                base = circuit.edge_weights_yy[r, idx]
                mod = ising_mod[:, r, 2, edge_idx]
                angle = circuit.eta[r, 2] * base * (1 + mod)
                state = apply_two_qubit_gate(
                    state, ising_yy_matrix(angle), i, j, circuit.n_qubits)
            nb_y = circuit.node_biases_y[r, :].expand(circuit.n_qubits)
            angles_y = circuit.eta[r, 2] * nb_y.unsqueeze(0).expand(batch_size, -1)
            ry_bias = ry_matrix(angles_y)
            for v in range(circuit.n_qubits):
                state = apply_single_qubit_gate(state, ry_bias[:, v], v, circuit.n_qubits)
            state = circuit._apply_trainable_rotations(state, r, 2)
            if r < circuit.R - 1:
                state = circuit._apply_cnot_ladder(state)

        probs = state.real ** 2 + state.imag ** 2
        return torch.einsum('bd,qd->bq', probs, circuit._z_signs)
