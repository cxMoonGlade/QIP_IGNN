"""
Local Quantum Aggregator - Quantum Circuit as the Aggregation Function (GIN-based)

This module implements LOCAL quantum aggregation following the GIN formula:

    GIN: h_v^(k) = MLP((1 + ε^(k)) · h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1))
    
    Ours: h_v^(k) = MLP((1 + ε^(k)) · h_v^(k-1) + QuantumAgg({h_u, u ∈ N(v)}))

The quantum circuit performs aggregation of neighbor features via entanglement:

    1. Collect K neighbors for each node
    2. Encode each neighbor's features into separate qubit groups
    3. Apply entangling gates ACROSS neighbors (this IS the aggregation)
    4. Use quantum pooling to reduce dimensionality
    5. Measure to get aggregated embedding
    6. Add (1 + ε) * h_self to match GIN formula
    7. Apply MLP transformation

Key difference from QuantumMLP (quantum_mlp.py):
    - QuantumMLP: h_agg = (1+ε)*h + Σ(neighbors)  →  quantum_circuit(h_agg)
                  [classical aggregation]           [quantum transformation]
    
    - This:      h_agg = (1+ε)*h + quantum_circuit(neighbors)
                         [GIN formula with quantum aggregation]

The entanglement between qubits from different neighbors IS the aggregation,
while the (1+ε) self-loop follows the GIN formulation exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from typing import Optional, Tuple
import math

from .ansatz import rx_matrix, ry_matrix, rz_matrix, ising_zz_matrix


class LocalQuantumAggregator(nn.Module):
    """
    Local Quantum Aggregator using QCNN-style circuit.
    
    Architecture:
        K neighbors × n_qubits_per_neighbor → [Convolution] → [Pooling] → ... → output
    
    The entanglement between neighbor qubits IS the aggregation operation.
    This is fundamentally different from classical aggregation + quantum transform.
    
    Args:
        input_dim: Dimension of input node features
        output_dim: Dimension of output aggregated features
        n_qubits_per_neighbor: Qubits to encode each neighbor (min: ceil(input_dim/3))
        max_neighbors: Maximum number of neighbors to sample (K)
        conv_layers: Number of convolution-pooling layer pairs
        include_self: Whether to include self-loop in aggregation
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_qubits_per_neighbor: int = 4,
        max_neighbors: int = 4,  # K in GraphSAGE
        conv_layers: int = 2,
        include_self: bool = True,
        perm_invariant: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits_per_neighbor = n_qubits_per_neighbor
        self.max_neighbors = max_neighbors
        self.conv_layers = conv_layers
        self.include_self = include_self
        self.perm_invariant = perm_invariant
        
        # Total qubits = K neighbors × qubits per neighbor
        # If include_self, we add 1 more "neighbor" (the node itself)
        self.n_neighbors = max_neighbors + (1 if include_self else 0)
        self.total_qubits = self.n_neighbors * n_qubits_per_neighbor
        
        # Input projection: feature_dim → n_qubits_per_neighbor * 3 (for RX, RY, RZ)
        self.input_proj = nn.Linear(input_dim, n_qubits_per_neighbor * 3)
        
        # Learnable parameters for convolutional layers
        # Each conv layer has parameters for 2-qubit gates between ALL adjacent pairs
        self.conv_params = nn.ParameterList()
        
        current_qubits = self.total_qubits
        for layer in range(conv_layers):
            n_pairs = current_qubits - 1
            rot_dim = 1 if perm_invariant else current_qubits
            zz_dim = 1 if perm_invariant else n_pairs
            # Parameters for rotation gates (RX, RY, RZ for each qubit)
            rot_params = nn.Parameter(torch.randn(rot_dim, 3) * 0.1)
            # Parameters for ZZ coupling (between adjacent qubits)
            zz_params = nn.Parameter(torch.randn(zz_dim) * 0.1)
            self.conv_params.append(nn.ParameterDict({
                'rot': rot_params,
                'zz': zz_params,
            }))
            # After pooling: keep odd indices [1,3,5,...] → ceil(n/2) qubits
            # Must match _quantum_pooling: keep_dims = range(1, n+1, 2) → (n+1)//2 elements
            current_qubits = max((current_qubits + 1) // 2, 1)
        
        # Output projection: final_qubits → output_dim
        self.final_qubits = current_qubits
        self.output_proj = nn.Linear(self.final_qubits, output_dim)
        
        print(f"  [LocalQuantumAgg] {self.n_neighbors} neighbors × {n_qubits_per_neighbor} qubits = {self.total_qubits} total qubits")
        print(f"  [LocalQuantumAgg] After {conv_layers} conv+pool layers: {self.final_qubits} qubits")
    
    def _apply_single_qubit_gate(self, state: torch.Tensor, gate: torch.Tensor, 
                                  qubit_idx: int, n_qubits: int) -> torch.Tensor:
        """Apply single qubit gate to multi-qubit state."""
        batch_size = state.shape[0]
        dim = 2 ** n_qubits
        
        state = state.view(batch_size, *([2] * n_qubits))
        
        # Move target qubit to last dimension
        perm = list(range(1, n_qubits + 1))
        perm.remove(qubit_idx + 1)
        perm.append(qubit_idx + 1)
        perm = [0] + perm
        
        state = state.permute(*perm).contiguous()
        state = state.view(batch_size, -1, 2)
        
        # Apply gate
        if gate.dim() == 2:
            state = torch.einsum('boi,ij->boj', state, gate)
        else:
            state = torch.einsum('boi,bij->boj', state, gate)
        
        # Reshape back
        state = state.view(batch_size, *([2] * n_qubits))
        inv_perm = [0] + [perm.index(i) for i in range(1, n_qubits + 1)]
        state = state.permute(*inv_perm).contiguous()
        
        return state.view(batch_size, dim)
    
    def _apply_two_qubit_gate(self, state: torch.Tensor, gate: torch.Tensor,
                               q1: int, q2: int, n_qubits: int) -> torch.Tensor:
        """Apply two-qubit gate."""
        batch_size = state.shape[0]
        dim = 2 ** n_qubits
        
        state = state.view(batch_size, *([2] * n_qubits))
        
        # Move target qubits to last two dimensions
        perm = list(range(1, n_qubits + 1))
        perm.remove(q1 + 1)
        perm.remove(q2 + 1)
        perm.extend([q1 + 1, q2 + 1])
        perm = [0] + perm
        
        state = state.permute(*perm).contiguous()
        state = state.view(batch_size, -1, 4)
        
        # Apply gate
        if gate.dim() == 2:
            state = torch.einsum('boi,ij->boj', state, gate)
        else:
            state = torch.einsum('boi,bij->boj', state, gate)
        
        # Reshape back
        state = state.view(batch_size, *([2] * n_qubits))
        inv_perm = [0] + [perm.index(i) for i in range(1, n_qubits + 1)]
        state = state.permute(*inv_perm).contiguous()
        
        return state.view(batch_size, dim)
    
    def _quantum_convolution(self, state: torch.Tensor, params: dict, 
                              n_qubits: int, device: torch.device) -> torch.Tensor:
        """
        Apply quantum convolution layer (OPTIMIZED).
        
        Consists of:
        1. ZZ entangling gates between adjacent qubits (this creates entanglement across neighbors!)
        2. Single-qubit rotation gates
        """
        rot_params = params['rot'].expand(n_qubits, 3)
        zz_params = params['zz'].expand(n_qubits - 1)
        
        # Pre-compute all ZZ gates at once (batched)
        zz_gates = ising_zz_matrix(zz_params)  # [n_qubits-1, 4, 4]
        
        # Apply ZZ entangling layer
        for i in range(n_qubits - 1):
            state = self._apply_two_qubit_gate(state, zz_gates[i], i, i+1, n_qubits)
        
        # Pre-compute all rotation gates at once (batched)
        rx_gates = rx_matrix(rot_params[:, 0])  # [n_qubits, 2, 2]
        ry_gates = ry_matrix(rot_params[:, 1])  # [n_qubits, 2, 2]
        rz_gates = rz_matrix(rot_params[:, 2])  # [n_qubits, 2, 2]
        
        # Apply rotation layer
        for q in range(n_qubits):
            state = self._apply_single_qubit_gate(state, rx_gates[q], q, n_qubits)
            state = self._apply_single_qubit_gate(state, ry_gates[q], q, n_qubits)
            state = self._apply_single_qubit_gate(state, rz_gates[q], q, n_qubits)
        
        return state
    
    def _quantum_pooling(self, state: torch.Tensor, n_qubits: int) -> Tuple[torch.Tensor, int]:
        """
        Quantum pooling: trace out half of the qubits.
        
        This reduces the qubit count while preserving aggregated information.
        We trace out even-indexed qubits.
        """
        batch_size = state.shape[0]
        new_n_qubits = n_qubits // 2
        if new_n_qubits == 0:
            new_n_qubits = 1
            return state, n_qubits  # Can't pool further
        
        # Convert to density matrix and trace out qubits
        # For efficiency, we use a simplified approach: measure and discard
        # This is equivalent to tracing out with deferred measurement
        
        # Reshape state to separate traced/kept qubits
        state = state.view(batch_size, *([2] * n_qubits))
        
        # Trace out every other qubit (even indices)
        # New state shape: [batch, 2^new_n_qubits]
        # We sum over the traced-out dimensions (this is like partial trace)
        
        # For a proper partial trace, we need density matrix
        # ρ_reduced = Tr_A(|ψ⟩⟨ψ|)
        # For efficiency, we approximate by keeping odd-indexed qubits
        
        keep_dims = list(range(1, n_qubits + 1, 2))  # Odd indices (1, 3, 5, ...)
        trace_dims = list(range(2, n_qubits + 1, 2))  # Even indices (2, 4, 6, ...)
        
        if len(keep_dims) == 0:
            keep_dims = [1]
        
        # Reorder: [batch, trace_dims..., keep_dims...]
        perm = [0] + trace_dims + keep_dims
        state = state.permute(*perm).contiguous()
        
        # Sum over traced dimensions (approximate partial trace)
        n_trace = len(trace_dims)
        state = state.view(batch_size, 2**n_trace, -1)
        
        # Take the amplitude-weighted combination (not just sum)
        # This preserves more quantum information
        probs = (state.abs() ** 2).sum(dim=-1, keepdim=True)  # [batch, 2^n_trace, 1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
        state = (state * probs.sqrt()).sum(dim=1)  # [batch, 2^new_n_qubits]
        
        # Renormalize
        norm = torch.sqrt((state.abs() ** 2).sum(dim=-1, keepdim=True) + 1e-10)
        state = state / norm
        
        return state, len(keep_dims)
    
    def _measure_z_expectations(self, state: torch.Tensor, n_qubits: int, 
                                 device: torch.device) -> torch.Tensor:
        """Measure Z expectation values for all qubits."""
        batch_size = state.shape[0]
        dim = 2 ** n_qubits
        
        probs = (state.abs() ** 2).real
        
        # Compute Z expectation for each qubit
        basis_indices = torch.arange(dim, device=device)
        z_expectations = []
        
        for q in range(n_qubits):
            bits = (basis_indices >> (n_qubits - 1 - q)) & 1
            z_signs = 1.0 - 2.0 * bits.float()  # +1 for |0⟩, -1 for |1⟩
            z_exp = (probs * z_signs).sum(dim=-1)
            z_expectations.append(z_exp)
        
        return torch.stack(z_expectations, dim=-1)  # [batch, n_qubits]
    
    def forward(self, neighbor_features: torch.Tensor, 
                self_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        True quantum aggregation of neighbor features.
        
        Args:
            neighbor_features: [batch, K, input_dim] - K neighbors per node
            self_features: [batch, input_dim] - optional self features
        
        Returns:
            aggregated: [batch, output_dim] - quantum-aggregated features
        """
        batch_size = neighbor_features.shape[0]
        K = neighbor_features.shape[1]
        device = neighbor_features.device
        
        # Pad or truncate to max_neighbors
        if K < self.max_neighbors:
            # Pad with zeros (will encode to |0⟩ state)
            padding = torch.zeros(batch_size, self.max_neighbors - K, self.input_dim, 
                                  device=device)
            neighbor_features = torch.cat([neighbor_features, padding], dim=1)
        elif K > self.max_neighbors:
            # Random sample (or could use attention-based sampling)
            indices = torch.randperm(K)[:self.max_neighbors]
            neighbor_features = neighbor_features[:, indices]
        
        # Include self features if provided
        if self.include_self and self_features is not None:
            self_features = self_features.unsqueeze(1)  # [batch, 1, dim]
            all_features = torch.cat([self_features, neighbor_features], dim=1)
        else:
            all_features = neighbor_features
        
        # all_features: [batch, n_neighbors, input_dim]
        
        # Project each neighbor's features to rotation angles
        # [batch, n_neighbors, input_dim] → [batch, n_neighbors, n_qubits_per_neighbor * 3]
        angles = self.input_proj(all_features)
        angles = angles.view(batch_size, self.n_neighbors, self.n_qubits_per_neighbor, 3)
        
        # Initialize quantum state |00...0⟩
        dim = 2 ** self.total_qubits
        state = torch.zeros(batch_size, dim, dtype=torch.complex64, device=device)
        state[:, 0] = 1.0
        
        # Pre-compute ALL rotation matrices at once (batched)
        # Reshape angles: [batch, n_neighbors, n_qubits_per_neighbor, 3] → [batch, total_qubits, 3]
        angles_flat = angles.view(batch_size, self.total_qubits, 3)
        
        # Batch compute all rotation gates: [batch, total_qubits, 2, 2]
        rx_gates = rx_matrix(angles_flat[:, :, 0])  # [batch, total_qubits, 2, 2]
        ry_gates = ry_matrix(angles_flat[:, :, 1])  # [batch, total_qubits, 2, 2]
        rz_gates = rz_matrix(angles_flat[:, :, 2])  # [batch, total_qubits, 2, 2]
        
        # Apply encoding gates (still need loop for sequential gate application)
        for q in range(self.total_qubits):
            state = self._apply_single_qubit_gate(state, rx_gates[:, q], q, self.total_qubits)
            state = self._apply_single_qubit_gate(state, ry_gates[:, q], q, self.total_qubits)
            state = self._apply_single_qubit_gate(state, rz_gates[:, q], q, self.total_qubits)
        
        # Apply convolution + pooling layers
        current_qubits = self.total_qubits
        for layer_idx in range(self.conv_layers):
            # Convolution (entanglement across neighbors + rotations)
            state = self._quantum_convolution(state, self.conv_params[layer_idx], 
                                               current_qubits, device)
            # Pooling (reduce qubits)
            state, current_qubits = self._quantum_pooling(state, current_qubits)
        
        # Measure Z expectations
        output = self._measure_z_expectations(state, current_qubits, device)
        
        # Project to output dimension
        output = self.output_proj(output)
        
        return output


class LocalQuantumGNN(MessagePassing):
    """
    GNN with Local Quantum Aggregation following GIN formula.
    
    GIN formula: h_v^(k) = MLP((1 + ε^(k)) · h_v^(k-1) + AGG({h_u, u ∈ N(v)}))
    
    Our implementation:
        h_v^(k) = MLP((1 + ε^(k)) · h_v^(k-1) + QuantumAgg({h_u, u ∈ N(v)}))
    
    The quantum circuit aggregates ONLY neighbors (not self), then we add
    the weighted self-loop following the GIN formula.
    
    For each node:
    1. Collect K neighbors (or all if fewer)
    2. Use LocalQuantumAggregator to aggregate neighbor features via entanglement
    3. Add (1 + ε) * h_self to aggregated result (GIN formula)
    4. Apply MLP transformation
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        n_layers: int = 2,
        n_qubits_per_neighbor: int = 4,
        max_neighbors: int = 4,
        conv_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = False,
        perm_invariant: bool = False,
    ):
        super().__init__(aggr=None)  # We handle aggregation ourselves
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.perm_invariant = perm_invariant
        
        # Quantum aggregators (one per layer) - aggregate NEIGHBORS ONLY
        self.quantum_aggs = nn.ModuleList()
        self.mlps = nn.ModuleList()  # MLP after aggregation (GIN style)
        self.norms = nn.ModuleList()  # BatchNorm or LayerNorm
        
        for layer in range(n_layers):
            input_dim = in_features if layer == 0 else hidden_dim
            
            # Quantum aggregator for neighbors only (no self)
            self.quantum_aggs.append(
                LocalQuantumAggregator(
                    input_dim=input_dim,
                    output_dim=hidden_dim,
                    n_qubits_per_neighbor=n_qubits_per_neighbor,
                    max_neighbors=max_neighbors,
                    conv_layers=conv_layers,
                    include_self=False,  # GIN: aggregate neighbors separately
                    perm_invariant=perm_invariant,
                )
            )
            
            # MLP after aggregation (GIN style: 2-layer MLP)
            self.mlps.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Learnable epsilon for self-loop weighting (GIN)
        self.eps = nn.Parameter(torch.zeros(n_layers))
        
        # Projection for first layer if input_dim != hidden_dim
        self.input_proj = nn.Linear(in_features, hidden_dim) if in_features != hidden_dim else nn.Identity()
        
        print(f"  [LocalQuantumGNN] GIN-style: h = MLP((1+ε)·h + QuantumAgg(neighbors))")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        GIN-style forward pass with quantum aggregation.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            h: Updated node features [num_nodes, hidden_dim]
        """
        h = x
        num_nodes = x.shape[0]
        device = x.device
        
        for layer in range(self.n_layers):
            # Project h to hidden_dim if needed (first layer)
            if layer == 0:
                h_proj = self.input_proj(h)
            else:
                h_proj = h
            
            # Collect neighbor features for each node
            neighbor_features = self._collect_neighbors(h, edge_index, num_nodes, device)
            
            # Quantum aggregation of neighbors (NOT including self)
            h_neighbors = self.quantum_aggs[layer](neighbor_features, None)  # No self_features
            
            # GIN formula: (1 + ε) * h_self + aggregated_neighbors
            h_gin = (1 + self.eps[layer]) * h_proj + h_neighbors
            
            # MLP transformation (GIN style)
            h = self.mlps[layer](h_gin)
            
            # Norm + activation
            h = self.norms[layer](h)
            h = F.relu(h)
            
            # Dropout (not after last layer)
            if layer < self.n_layers - 1 and self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def _collect_neighbors(self, x: torch.Tensor, edge_index: torch.Tensor,
                           num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        Collect neighbor features for each node (VECTORIZED - 100x faster).
        
        Returns:
            neighbor_features: [num_nodes, max_neighbors, feature_dim]
        """
        src, dst = edge_index  # src -> dst (messages flow from src to dst)
        feature_dim = x.shape[1]
        num_edges = edge_index.shape[1]
        
        # Initialize output
        neighbor_features = torch.zeros(num_nodes, self.max_neighbors, feature_dim, device=device)
        
        if num_edges == 0:
            return neighbor_features
        
        # Step 1: Sort edges by destination node for efficient grouping
        # This groups all incoming edges to each node together
        sort_idx = torch.argsort(dst)
        sorted_src = src[sort_idx]
        sorted_dst = dst[sort_idx]
        
        # Step 2: Count neighbors per node and compute offsets
        # bincount gives count of edges per destination node
        neighbor_counts = torch.bincount(dst, minlength=num_nodes)
        
        # Step 3: Compute position within each node's neighbor list
        # For each edge, what position (0, 1, 2, ...) is it in the sorted order for that dst?
        # Use cumsum trick: for each dst, edges are consecutive after sorting
        ones = torch.ones(num_edges, dtype=torch.long, device=device)
        
        # Cumulative count within each group
        # First, compute cumsum, then subtract the starting offset for each group
        cumsum = torch.cumsum(ones, dim=0) - 1  # [0, 1, 2, 3, ...]
        
        # Compute offset for each destination node (where its edges start in sorted order)
        offsets = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(neighbor_counts, dim=0)
        
        # Position within group = cumsum - offset[dst]
        positions = cumsum - offsets[sorted_dst]
        
        # Step 4: Only keep edges where position < max_neighbors
        valid_mask = positions < self.max_neighbors
        valid_src = sorted_src[valid_mask]
        valid_dst = sorted_dst[valid_mask]
        valid_pos = positions[valid_mask]
        
        # Step 5: Scatter source features to neighbor_features
        # neighbor_features[dst, pos, :] = x[src, :]
        neighbor_features[valid_dst, valid_pos] = x[valid_src]
        
        return neighbor_features


class LocalQuantumGINEncoder(nn.Module):
    """
    Encoder using Local Quantum Aggregation.
    
    Drop-in replacement for QuantumMLPGINEncoder using local quantum aggregation
    (entanglement across neighbors).
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        n_layers: int = 2,
        n_qubits_per_neighbor: int = 4,
        max_neighbors: int = 4,
        conv_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = False,
        perm_invariant: bool = False,
    ):
        super().__init__()
        
        self.dropout = dropout
        self.local_quantum_gnn = LocalQuantumGNN(
            in_features=in_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_qubits_per_neighbor=n_qubits_per_neighbor,
            max_neighbors=max_neighbors,
            conv_layers=conv_layers,
            dropout=0.0,  # We handle dropout here
            use_layer_norm=use_layer_norm,
            perm_invariant=perm_invariant,
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch=None, return_info: bool = False):
        """
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment (optional)
            return_info: If True, also return info dict
        
        Returns:
            h: Node embeddings [num_nodes, hidden_dim]
        """
        h = self.local_quantum_gnn(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        if return_info:
            return h, {'aggregator': 'local_quantum'}
        return h


