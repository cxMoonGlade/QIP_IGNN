"""
Deep **XYZ** parametrized quantum circuit (paper appendix *Deep XYZ Ansatz Used in
Experiments*): alternating data-dependent encodings (angle / re-uploading blocks),
Ising-type *ZZ* / *XX* / *YY* couplers, and trainable single-qubit rotations
:math:`U(\\xi)`.  Used by :class:`qignn.quantum_torch.TorchQuantumLayer` /
:class:`qignn.quantum_torch.TopoAwareQuantumLayer` as the trainable *U(ξ)* in the
encode--unitary--measure definition of the quantum residual.  All simulation is
PyTorch statevector math on GPU/CPU.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


# =============================================================================
# Quantum Gates as PyTorch Operations
# =============================================================================

def rx_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RX rotation gate: exp(-i * theta/2 * X)."""
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    zeros = torch.zeros_like(theta)

    real = torch.stack([
        torch.stack([cos, zeros], dim=-1),
        torch.stack([zeros, cos], dim=-1)
    ], dim=-2)

    imag = torch.stack([
        torch.stack([zeros, -sin], dim=-1),
        torch.stack([-sin, zeros], dim=-1)
    ], dim=-2)

    return torch.complex(real, imag)


def ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RY rotation gate: exp(-i * theta/2 * Y)."""
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)

    real = torch.stack([
        torch.stack([cos, -sin], dim=-1),
        torch.stack([sin, cos], dim=-1)
    ], dim=-2)

    imag = torch.zeros_like(real)
    return torch.complex(real, imag)


def rz_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RZ rotation gate: exp(-i * theta/2 * Z)."""
    phase = theta / 2
    zeros = torch.zeros_like(theta)

    real = torch.stack([
        torch.stack([torch.cos(-phase), zeros], dim=-1),
        torch.stack([zeros, torch.cos(phase)], dim=-1)
    ], dim=-2)

    imag = torch.stack([
        torch.stack([torch.sin(-phase), zeros], dim=-1),
        torch.stack([zeros, torch.sin(phase)], dim=-1)
    ], dim=-2)

    return torch.complex(real, imag)


def cnot_matrix(device: torch.device) -> torch.Tensor:
    """CNOT gate (control on first qubit)."""
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.complex64, device=device)


def ising_zz_matrix(theta: torch.Tensor) -> torch.Tensor:
    """IsingZZ interaction gate: exp(-i * theta/2 * Z otimes Z)."""
    phase = theta / 2

    diag_real = torch.stack([
        torch.cos(-phase),
        torch.cos(phase),
        torch.cos(phase),
        torch.cos(-phase),
    ], dim=-1)

    diag_imag = torch.stack([
        torch.sin(-phase),
        torch.sin(phase),
        torch.sin(phase),
        torch.sin(-phase),
    ], dim=-1)

    batch_shape = theta.shape
    eye = torch.eye(4, device=theta.device, dtype=theta.dtype)
    eye = eye.expand(*batch_shape, 4, 4).clone()

    real = eye * diag_real.unsqueeze(-1)
    imag = eye * diag_imag.unsqueeze(-1)

    return torch.complex(real, imag)


def ising_xx_matrix(theta: torch.Tensor) -> torch.Tensor:
    """IsingXX interaction gate: exp(-i * theta/2 * X otimes X)."""
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)

    real = torch.zeros(*theta.shape, 4, 4, device=theta.device, dtype=theta.dtype)
    imag = torch.zeros(*theta.shape, 4, 4, device=theta.device, dtype=theta.dtype)

    real[..., 0, 0] = cos
    real[..., 1, 1] = cos
    real[..., 2, 2] = cos
    real[..., 3, 3] = cos

    imag[..., 0, 3] = -sin
    imag[..., 1, 2] = -sin
    imag[..., 2, 1] = -sin
    imag[..., 3, 0] = -sin

    return torch.complex(real, imag)


def ising_yy_matrix(theta: torch.Tensor) -> torch.Tensor:
    """IsingYY interaction gate: exp(-i * theta/2 * Y otimes Y)."""
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)

    real = torch.zeros(*theta.shape, 4, 4, device=theta.device, dtype=theta.dtype)
    imag = torch.zeros(*theta.shape, 4, 4, device=theta.device, dtype=theta.dtype)

    real[..., 0, 0] = cos
    real[..., 1, 1] = cos
    real[..., 2, 2] = cos
    real[..., 3, 3] = cos

    imag[..., 0, 3] = sin
    imag[..., 1, 2] = -sin
    imag[..., 2, 1] = -sin
    imag[..., 3, 0] = sin

    return torch.complex(real, imag)


# =============================================================================
# State Vector Operations
# =============================================================================

def apply_single_qubit_gate(
    state: torch.Tensor, gate: torch.Tensor, qubit: int, n_qubits: int
) -> torch.Tensor:
    """
    Apply a single-qubit gate to a batched state vector.

    Args:
        state: [batch, 2^n_qubits] complex
        gate:  [batch, 2, 2] complex
        qubit: target qubit index (0-indexed, MSB ordering)
        n_qubits: total qubit count
    """
    batch_size = state.shape[0]
    dim = 2 ** n_qubits

    left_dim = 2 ** (n_qubits - qubit - 1)
    right_dim = 2 ** qubit

    state_reshaped = state.view(batch_size, left_dim, 2, right_dim)
    new_state = torch.einsum('bij,bljr->blir', gate, state_reshaped)

    return new_state.reshape(batch_size, dim)


def apply_two_qubit_gate(
    state: torch.Tensor, gate: torch.Tensor,
    qubit1: int, qubit2: int, n_qubits: int
) -> torch.Tensor:
    """
    Apply a two-qubit gate to a batched state vector.

    Args:
        state: [batch, 2^n_qubits] complex
        gate:  [batch, 4, 4] or [4, 4] complex
        qubit1, qubit2: target qubit indices
        n_qubits: total qubit count
    """
    batch_size = state.shape[0]
    dim = 2 ** n_qubits

    if qubit1 > qubit2:
        qubit1, qubit2 = qubit2, qubit1

    state_reshaped = state.view(batch_size, *([2] * n_qubits))

    q1_pos = qubit1
    q2_pos = qubit2

    other_dims = [i for i in range(n_qubits) if i != q1_pos and i != q2_pos]
    perm = [0] + [d + 1 for d in other_dims] + [q1_pos + 1, q2_pos + 1]

    state_perm = state_reshaped.permute(*perm)

    other_size = 2 ** (n_qubits - 2)
    state_flat = state_perm.reshape(batch_size, other_size, 4)

    if gate.dim() == 2:
        new_state = torch.einsum('ij,bkj->bki', gate, state_flat)
    else:
        new_state = torch.einsum('bij,bkj->bki', gate, state_flat)

    new_state = new_state.reshape(batch_size, *([2] * (n_qubits - 2)), 2, 2)

    inv_perm = [0] + [0] * n_qubits
    for i, p in enumerate(perm[1:]):
        inv_perm[p] = i + 1

    new_state = new_state.permute(*inv_perm).contiguous()
    return new_state.reshape(batch_size, dim)


# =============================================================================
# Qubit Connectivity Topologies
# =============================================================================

def get_edge_pairs(n_qubits: int, topology: str = 'linear') -> List[Tuple[int, int]]:
    """
    Get edge pairs for quantum circuit entanglement topology.

    Args:
        n_qubits: Number of qubits
        topology: 'linear', 'circular', or 'all'
    """
    if topology == 'linear':
        return [(i, i + 1) for i in range(n_qubits - 1)]
    elif topology == 'circular':
        return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    elif topology == 'all':
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    else:
        raise ValueError(f"Unknown topology: {topology}")


# =============================================================================
# Deep XYZ Quantum Circuit Ansatz
# =============================================================================

class TorchQuantumCircuit(nn.Module):
    """
    Pure PyTorch Deep XYZ quantum circuit matching PennyLane structure.

    3 blocks per rep: ZZ, XX, YY.
    Each block: data encode -> Ising interaction -> node bias -> trainable rots -> CNOT.
    Supports learnable encoding scale/bias and graph-conditioned edge weights.
    """

    def __init__(
        self,
        n_qubits: int,
        edge_pairs: List[Tuple[int, int]],
        circuit_reps: int = 1,
        graph_conditioned: bool = False,
        hidden_dim: int = 64,
        perm_invariant: bool = False,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.edge_pairs = edge_pairs
        self.n_edges = len(edge_pairs)
        self.R = circuit_reps
        self.graph_conditioned = graph_conditioned
        self.perm_invariant = perm_invariant
        self.dim = 2 ** n_qubits

        self._init_parameters()

        if graph_conditioned:
            n_edge_params = circuit_reps * 3 * self.n_edges
            self.edge_weight_net = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, n_edge_params),
                nn.Tanh(),
            )

        self.register_buffer('_cnot', cnot_matrix(torch.device('cpu')))

        z_signs = torch.ones(n_qubits, self.dim)
        for i in range(n_qubits):
            for k in range(self.dim):
                if (k >> (n_qubits - i - 1)) & 1:
                    z_signs[i, k] = -1
        self.register_buffer('_z_signs', z_signs)

    def _init_parameters(self):
        n = self.n_qubits
        R = self.R
        n_edges = self.n_edges
        pi = self.perm_invariant

        q_dim = 1 if pi else n
        e_dim = 1 if pi else n_edges

        self.encoding_scale = nn.Parameter(
            torch.ones(q_dim, 3) * np.pi + torch.randn(q_dim, 3) * 0.1)
        self.encoding_bias = nn.Parameter(torch.randn(q_dim, 3) * 0.1)

        self.eta = nn.Parameter(torch.ones(R, 3) * 0.5)

        self.edge_weights_zz = nn.Parameter(
            torch.ones(R, e_dim) + torch.randn(R, e_dim) * 0.1)
        self.edge_weights_xx = nn.Parameter(
            torch.ones(R, e_dim) + torch.randn(R, e_dim) * 0.1)
        self.edge_weights_yy = nn.Parameter(
            torch.ones(R, e_dim) + torch.randn(R, e_dim) * 0.1)

        self.node_biases_z = nn.Parameter(torch.randn(R, q_dim) * 0.1)
        self.node_biases_x = nn.Parameter(torch.randn(R, q_dim) * 0.1)
        self.node_biases_y = nn.Parameter(torch.randn(R, q_dim) * 0.1)

        self.trainable_rots = nn.Parameter(torch.randn(R, 3, q_dim, 3) * 0.1)

    def _init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        state = torch.zeros(batch_size, self.dim, dtype=torch.complex64, device=device)
        state[:, 0] = 1.0
        return state

    def _precompute_encoding_gates(self, data: torch.Tensor):
        scale = self.encoding_scale.expand(self.n_qubits, 3)
        bias = self.encoding_bias.expand(self.n_qubits, 3)
        angles_x = scale[:, 0] * data + bias[:, 0]
        angles_y = scale[:, 1] * data + bias[:, 1]
        angles_z = scale[:, 2] * data + bias[:, 2]

        return rx_matrix(angles_x), ry_matrix(angles_y), rz_matrix(angles_z)

    def _apply_data_encoding_cached(
        self, state, rx_gates, ry_gates, rz_gates, block_type='full'
    ):
        for q in range(self.n_qubits):
            if block_type == 'full':
                state = apply_single_qubit_gate(state, rx_gates[:, q], q, self.n_qubits)
                state = apply_single_qubit_gate(state, ry_gates[:, q], q, self.n_qubits)
                state = apply_single_qubit_gate(state, rz_gates[:, q], q, self.n_qubits)
            elif block_type == 'xy':
                state = apply_single_qubit_gate(state, rx_gates[:, q], q, self.n_qubits)
                state = apply_single_qubit_gate(state, ry_gates[:, q], q, self.n_qubits)
            elif block_type == 'yz':
                state = apply_single_qubit_gate(state, ry_gates[:, q], q, self.n_qubits)
                state = apply_single_qubit_gate(state, rz_gates[:, q], q, self.n_qubits)
        return state

    def _apply_trainable_rotations(self, state, r, block):
        batch_size = state.shape[0]
        n = self.n_qubits

        rots = self.trainable_rots[r, block]  # [q_dim, 3]
        rots = rots.expand(n, 3)  # broadcast if perm_invariant (1->n)
        angles_x = rots[:, 0].unsqueeze(0).expand(batch_size, -1)
        angles_y = rots[:, 1].unsqueeze(0).expand(batch_size, -1)
        angles_z = rots[:, 2].unsqueeze(0).expand(batch_size, -1)

        rx_g = rx_matrix(angles_x)
        ry_g = ry_matrix(angles_y)
        rz_g = rz_matrix(angles_z)

        for q in range(self.n_qubits):
            state = apply_single_qubit_gate(state, rx_g[:, q], q, self.n_qubits)
            state = apply_single_qubit_gate(state, ry_g[:, q], q, self.n_qubits)
            state = apply_single_qubit_gate(state, rz_g[:, q], q, self.n_qubits)

        return state

    def _apply_cnot_ladder(self, state):
        cnot = self._cnot.to(state.device)
        for (i, j) in self.edge_pairs:
            state = apply_two_qubit_gate(state, cnot, i, j, self.n_qubits)
        return state

    def forward(
        self,
        data: torch.Tensor,
        graph_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Execute the quantum circuit.

        Args:
            data: [batch, n_qubits] input features
            graph_embedding: [batch, hidden_dim] optional graph conditioning

        Returns:
            Z expectation values [batch, n_qubits]
        """
        batch_size = data.shape[0]
        device = data.device

        state = self._init_state(batch_size, device)

        if self.graph_conditioned and graph_embedding is not None:
            dynamic_weights = self.edge_weight_net(graph_embedding)
            dynamic_weights = dynamic_weights.view(batch_size, self.R, 3, self.n_edges)
        else:
            dynamic_weights = None

        def get_edge_weights(r, ising_type, edge_idx):
            idx = 0 if self.perm_invariant else edge_idx
            if ising_type == 0:
                base_weight = self.edge_weights_zz[r, idx]
            elif ising_type == 1:
                base_weight = self.edge_weights_xx[r, idx]
            else:
                base_weight = self.edge_weights_yy[r, idx]
            if dynamic_weights is not None:
                return base_weight * (1 + dynamic_weights[:, r, ising_type, edge_idx])
            else:
                return base_weight.expand(batch_size)

        rx_gates, ry_gates, rz_gates = self._precompute_encoding_gates(data)

        for r in range(self.R):
            # Block 1: ZZ
            state = self._apply_data_encoding_cached(
                state, rx_gates, ry_gates, rz_gates, block_type='full')
            for edge_idx, (i, j) in enumerate(self.edge_pairs):
                angle = self.eta[r, 0] * get_edge_weights(r, 0, edge_idx)
                state = apply_two_qubit_gate(
                    state, ising_zz_matrix(angle), i, j, self.n_qubits)
            nb_z = self.node_biases_z[r, :].expand(self.n_qubits)
            angles_z = self.eta[r, 0] * nb_z.unsqueeze(0).expand(batch_size, -1)
            rz_bias = rz_matrix(angles_z)
            for v in range(self.n_qubits):
                state = apply_single_qubit_gate(state, rz_bias[:, v], v, self.n_qubits)
            state = self._apply_trainable_rotations(state, r, 0)
            state = self._apply_cnot_ladder(state)

            # Block 2: XX
            state = self._apply_data_encoding_cached(
                state, rx_gates, ry_gates, rz_gates, block_type='xy')
            for edge_idx, (i, j) in enumerate(self.edge_pairs):
                angle = self.eta[r, 1] * get_edge_weights(r, 1, edge_idx)
                state = apply_two_qubit_gate(
                    state, ising_xx_matrix(angle), i, j, self.n_qubits)
            nb_x = self.node_biases_x[r, :].expand(self.n_qubits)
            angles_x = self.eta[r, 1] * nb_x.unsqueeze(0).expand(batch_size, -1)
            rx_bias = rx_matrix(angles_x)
            for v in range(self.n_qubits):
                state = apply_single_qubit_gate(state, rx_bias[:, v], v, self.n_qubits)
            state = self._apply_trainable_rotations(state, r, 1)
            state = self._apply_cnot_ladder(state)

            # Block 3: YY
            state = self._apply_data_encoding_cached(
                state, rx_gates, ry_gates, rz_gates, block_type='yz')
            for edge_idx, (i, j) in enumerate(self.edge_pairs):
                angle = self.eta[r, 2] * get_edge_weights(r, 2, edge_idx)
                state = apply_two_qubit_gate(
                    state, ising_yy_matrix(angle), i, j, self.n_qubits)
            nb_y = self.node_biases_y[r, :].expand(self.n_qubits)
            angles_y = self.eta[r, 2] * nb_y.unsqueeze(0).expand(batch_size, -1)
            ry_bias = ry_matrix(angles_y)
            for v in range(self.n_qubits):
                state = apply_single_qubit_gate(state, ry_bias[:, v], v, self.n_qubits)
            state = self._apply_trainable_rotations(state, r, 2)
            if r < self.R - 1:
                state = self._apply_cnot_ladder(state)

        # Measure Z expectations (vectorized)
        probs = state.real ** 2 + state.imag ** 2
        expectations = torch.einsum('bd,qd->bq', probs, self._z_signs)

        return expectations
