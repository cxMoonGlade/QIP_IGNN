r"""
Model components for *Quantum Injection Pathways for Implicit Graph Neural Networks*.

The paper fixes a **shared classical backbone** (encoder :math:`H=\mathrm{Enc}(A,X_0)`,
propagation :math:`\tilde A`, and map :math:`h_\theta` with :math:`\tanh` nonlinearity)
and varies **only** how a quantum (encode--unitary--measure) signal couples to
:math:`\Phi_\theta` in :math:`Z^\star = \Phi_\theta(Z^\star;\tilde A, H)`.

- **IN (independent)**: static :math:`Q_{\mathrm{IN}}(H,\tau(A))` (``q_ind_node``,
  :class:`TopoAwareQIGNN` → ``quantum_node``) is computed once and enters the
  pre-activation with :math:`H\Omega^\top` (same role as the IGNN input-injection term).
- **SD (state-dependent)**: in-loop residual :math:`\alpha\,g_\xi(\cdot)` with
  :math:`g_\xi(Z)` (``--quantum_inside --qi_direct``).
- **BD (backbone-dependent)**: :math:`\alpha\,g_\xi(h_\theta(Z))` (``--quantum_inside``,
  no ``--qi_direct``), where the code passes the *post*-\ :math:`\tanh` node state
  into the same residual module as the backbone output at the current iterate.

The implicit layer :class:`BatchedImplicitCore` implements the fixed-point solve
(``torchdeq``/Anderson in training) over batched dense graphs. Normalized
:math:`\tilde A` uses symmetric normalization with self-loops, as in the paper
(Kipf--Welling form; :func:`pyg_to_batched_dense`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool

from .ansatz import get_edge_pairs
from .quantum_torch import TopoAwareQuantumLayer

try:
    from torchdeq import get_deq
    TORCHDEQ_AVAILABLE = True
except ImportError:
    TORCHDEQ_AVAILABLE = False


# =============================================================================
# Encoders
# =============================================================================

class GIN_MLP(nn.Module):
    """Multi-layer perceptron matching official GIN implementation."""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim,
                 use_layer_norm: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers < 1:
            raise ValueError('num_layers should be >= 1')
        elif num_layers == 1:
            self.linears.append(nn.Linear(input_dim, output_dim))
        else:
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                if use_layer_norm:
                    self.norms.append(nn.LayerNorm(hidden_dim))
                else:
                    self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.num_layers == 1:
            return self.linears[0](x)
        h = x
        for i in range(self.num_layers - 1):
            h = self.linears[i](h)
            h = self.norms[i](h)
            h = F.relu(h)
        return self.linears[-1](h)


class GINEncoder(nn.Module):
    """
    GIN Encoder matching official "How Powerful are Graph Neural Networks?" implementation.

    No dropout between GNN layers (only final dropout in classifier).
    Returns list of all layer representations for proper JK (Jumping Knowledge).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        n_layers: int = 5,
        dropout: float = 0.5,
        jk_mode: str = 'sum',
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.jk_mode = jk_mode
        self.dropout = dropout
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer in range(n_layers):
            in_dim = in_features if layer == 0 else hidden_dim
            self.mlps.append(
                GIN_MLP(2, in_dim, hidden_dim, hidden_dim,
                        use_layer_norm=use_layer_norm))
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.eps = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x, edge_index, batch=None):
        """Returns last layer output."""
        h = x
        for layer in range(self.n_layers):
            row, col = edge_index
            neighbor_sum = torch.zeros(h.size(0), h.size(1),
                                       device=h.device, dtype=h.dtype)
            neighbor_sum.index_add_(0, row, h[col])
            h_agg = (1 + self.eps[layer]) * h + neighbor_sum
            h = self.mlps[layer](h_agg)
            h = self.norms[layer](h)
            h = F.relu(h)
        return h

    def get_all_layers(self, x, edge_index, batch=None):
        """Get all layer representations for JK (Jumping Knowledge)."""
        hidden_rep = [x]
        h = x
        for layer in range(self.n_layers):
            row, col = edge_index
            neighbor_sum = torch.zeros(h.size(0), h.size(1),
                                       device=h.device, dtype=h.dtype)
            neighbor_sum.index_add_(0, row, h[col])
            h_agg = (1 + self.eps[layer]) * h + neighbor_sum
            h = self.mlps[layer](h_agg)
            h = self.norms[layer](h)
            h = F.relu(h)
            hidden_rep.append(h)
        return hidden_rep


class SimpleEncoder(nn.Module):
    """Minimal MLP encoder (no graph structure) for ablation studies."""

    def __init__(self, in_features: int, hidden_dim: int, n_layers: int = 1,
                 use_layer_norm: bool = False):
        super().__init__()

        def get_norm(dim):
            return nn.LayerNorm(dim) if use_layer_norm else nn.BatchNorm1d(dim)

        layers = [nn.Linear(in_features, hidden_dim), get_norm(hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), get_norm(hidden_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*layers)

    def forward(self, x, edge_index, batch=None):
        return self.encoder(x)

    def get_all_layers(self, x, edge_index, batch=None):
        return [x, self.encoder(x)]


class MinEncoder(nn.Module):
    """Minimal encoder: single Linear projection, no message passing (IGNN-style)."""

    def __init__(self, in_features: int, hidden_dim: int, use_relu: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, hidden_dim)
        self.use_relu = use_relu

    def forward(self, x, edge_index=None, batch=None):
        h = self.linear(x)
        return F.relu(h) if self.use_relu else h

    def get_all_layers(self, x, edge_index=None, batch=None):
        h = self.forward(x, edge_index, batch)
        return [x, h]


# =============================================================================
# Pooling
# =============================================================================

class BatchedGraphPooling(nn.Module):
    """
    Graph-level pooling for PyG batched graphs.
    Supports: sum, mean, max, concat, attention.
    """

    def __init__(self, hidden_dim: int, pooling: str = 'sum', n_heads: int = 4):
        super().__init__()
        self.pooling = pooling
        self.hidden_dim = hidden_dim

        if pooling == 'concat':
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif pooling == 'attention':
            self.n_heads = n_heads
            self.head_dim = hidden_dim // n_heads
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'sum':
            return global_add_pool(h, batch)
        elif self.pooling == 'mean':
            return global_mean_pool(h, batch)
        elif self.pooling == 'max':
            from torch_geometric.nn import global_max_pool
            return global_max_pool(h, batch)
        elif self.pooling == 'concat':
            from torch_geometric.nn import global_max_pool
            h_sum = global_add_pool(h, batch)
            h_mean = global_mean_pool(h, batch)
            h_max = global_max_pool(h, batch)
            return self.mlp(torch.cat([h_sum, h_mean, h_max], dim=-1))
        elif self.pooling == 'attention':
            return self._batched_attention_pool(h, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def _batched_attention_pool(self, h, batch):
        batch_size = batch.max().item() + 1
        outputs = []
        for b in range(batch_size):
            mask = (batch == b)
            h_b = h[mask]
            n_nodes = h_b.shape[0]

            Q = self.q_proj(h_b).view(n_nodes, self.n_heads, self.head_dim)
            K = self.k_proj(h_b).view(n_nodes, self.n_heads, self.head_dim)
            V = self.v_proj(h_b).view(n_nodes, self.n_heads, self.head_dim)

            scores = torch.einsum('nhd,mhd->nmh', Q, K) / (self.head_dim ** 0.5)
            att_weights = F.softmax(scores, dim=1)
            attended = torch.einsum('nmh,mhd->nhd', att_weights, V)

            head_out = attended.sum(dim=0)
            outputs.append(self.out_proj(head_out.view(-1)))
        return torch.stack(outputs, dim=0)


# =============================================================================
# Dense-graph helpers for implicit core
# =============================================================================

def pyg_to_batched_dense(data, hidden_features: torch.Tensor,
                         add_self_loops: bool = False,
                         normalize_adj: bool = False):
    """
    Convert PyG batched graph to padded dense tensors.

    normalize_adj: If True, build the symmetrically normalized adjacency with
        self-loops (Kipf--Welling / paper: *Ã* = *D̂*^{-1/2} (*A*+*I*) *D̂*^{-1/2})
        for the implicit map's propagation matrix.

    Returns:
        padded_h: [batch, max_nodes, hidden]
        padded_adj: [batch, max_nodes, max_nodes]
        mask: [batch, max_nodes]
        batch_size, max_nodes
    """
    device = hidden_features.device
    batch = data.batch
    edge_index = data.edge_index

    batch_size = batch.max().item() + 1
    nodes_per_graph = torch.bincount(batch)
    max_nodes = nodes_per_graph.max().item()
    hidden_dim = hidden_features.shape[1]

    padded_h = torch.zeros(batch_size, max_nodes, hidden_dim, device=device)
    padded_adj = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
    mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)

    node_offsets = torch.cat([
        torch.tensor([0], device=device), nodes_per_graph.cumsum(0)[:-1]])

    for b in range(batch_size):
        n_nodes = nodes_per_graph[b].item()
        offset = node_offsets[b].item()

        padded_h[b, :n_nodes] = hidden_features[offset:offset + n_nodes]

        graph_mask = (batch == b)
        edge_mask = graph_mask[edge_index[0]] & graph_mask[edge_index[1]]
        local_edges = edge_index[:, edge_mask] - offset
        padded_adj[b, local_edges[0], local_edges[1]] = 1.0

        if add_self_loops:
            idx = torch.arange(n_nodes, device=device)
            padded_adj[b, idx, idx] = 1.0

        if normalize_adj:
            A = padded_adj[b, :n_nodes, :n_nodes]

            # 1) Ensure symmetry for implicit operator
            A = torch.maximum(A, A.transpose(0, 1))

            # 2) Ensure self-loops for stable degree matrix
            A = A.clone()
            idx = torch.arange(n_nodes, device=device)
            A[idx, idx] = 1.0

            # 3) Symmetric normalization: D^{-1/2} A D^{-1/2}
            D = A.sum(dim=1).clamp(min=1e-6)
            d_inv_sqrt = D.pow(-0.5)
            A_norm = d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)

            padded_adj[b, :n_nodes, :n_nodes] = A_norm

        mask[b, :n_nodes] = True

    return padded_h, padded_adj, mask, batch_size, max_nodes


# =============================================================================
# Implicit Equilibrium Layer
# =============================================================================

class BatchedImplicitCore(nn.Module):
    r"""
    Batched implicit equilibrium map :math:`Z \mapsto \Phi_\theta(Z;\tilde A, H)`.

    The classical piece matches the paper's backbone
    :math:`h_\theta(Z)=\sigma(\tilde A Z W^\top + H\Omega^\top + \mathbf 1 b^\top)`
    with :math:`\sigma=\tanh`.  The third argument to :meth:`forward` (``q_features``)
    is the **broadcast row bias** in that pre-activation: for **IN**, this is
    :math:`Q_{\mathrm{IN}}(H,\tau(A))` (constant over solver iterates). For SD/BD,
    with ``quantum_inside=True``, the map adds
    :math:`\alpha\,g_\xi(\cdot)` **after** the :math:`\tanh` (see :meth:`_phi_step`):
    state-dependent if ``qi_direct`` uses current :math:`Z`, backbone-dependent
    if it uses the :math:`\tanh` output.

    :math:`W` is projected to control :math:`\lVert W\rVert_2` relative to
    :math:`\lVert\tilde A\rVert_2` (see :meth:`_get_kappa_target` / :meth:`_project_spectral_norm`)
    in the spirit of the IGNN contraction template in the paper.
    """

    def __init__(
        self,
        hidden_dim: int,
        kappa: float = 0.999,
        max_iter: int = 30,
        tol: float = 1e-6,
        damping: float = 0.5,
        solver: str = 'simple',
        use_dynamic_film: bool = False,
        quantum_inside: bool = False,
        qi_n_qubits: int = 4,
        qi_circuit_reps: int = 1,
        qi_alpha: float = 0.1,
        qi_topo: bool = False,
        qi_topo_node_dim: int = 21,
        qi_topo_graph_dim: int = 22,
        perm_invariant: bool = False,
        ignn_injection: bool = False,
        qi_direct: bool = False,
        qi_classical: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ignn_injection = ignn_injection
        self.kappa = kappa
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.solver = solver
        self.use_dynamic_film = use_dynamic_film
        self.quantum_inside = quantum_inside
        self.qi_direct = qi_direct
        self.qi_classical = qi_classical
        # Contraction budget: ||J_Φ|| ≤ κ + α < 1 (spectral norm on q_proj/q_out bounds L_g)
        contraction_budget = 1.0 - kappa
        if quantum_inside and qi_alpha > contraction_budget * 0.9:
            self.qi_alpha = contraction_budget * 0.9
            print(f"    [QI] alpha auto-adjusted: {qi_alpha} -> {self.qi_alpha:.4f} "
                  f"(budget: 1-kappa={contraction_budget:.4f})")
        else:
            self.qi_alpha = qi_alpha
        self.qi_topo = qi_topo

        self.W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W)
        self.Omega = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.Omega)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # For spectral norm estimation of W (power iteration state)
        self.register_buffer('_w_u', torch.randn(hidden_dim))
        self.register_buffer('_w_u_initialized', torch.tensor(False))

        if use_dynamic_film:
            self.film_scale = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.film_shift = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh())
        else:
            self.film_scale = None
            self.film_shift = None

        if quantum_inside:
            from torch.nn.utils import spectral_norm

            if qi_classical:
                self.qc_inside = nn.Sequential(
                    spectral_norm(nn.Linear(hidden_dim, qi_n_qubits)),
                    nn.Tanh(),
                    spectral_norm(nn.Linear(qi_n_qubits, hidden_dim)),
                )
                mode_str = "classical-matched"
            else:
                edge_pairs = get_edge_pairs(qi_n_qubits, 'linear')
                if qi_topo:
                    from .quantum_torch import TopoAwareQuantumLayer
                    self.qc_inside = TopoAwareQuantumLayer(
                        hidden_dim=hidden_dim,
                        n_qubits=qi_n_qubits,
                        edge_pairs=edge_pairs,
                        circuit_reps=qi_circuit_reps,
                        topo_node_dim=qi_topo_node_dim,
                        topo_graph_dim=qi_topo_graph_dim,
                        use_topo_encoding=True,
                        use_topo_ising=True,
                        use_competitive_gate=True,
                        perm_invariant=perm_invariant,
                        use_layer_norm=False,
                    )
                    mode_str = "topo-conditioned"
                else:
                    from .quantum_torch import TorchQuantumLayer
                    self.qc_inside = TorchQuantumLayer(
                        hidden_dim=hidden_dim,
                        n_qubits=qi_n_qubits,
                        edge_pairs=edge_pairs,
                        circuit_reps=qi_circuit_reps,
                        graph_conditioned=False,
                        perm_invariant=perm_invariant,
                        use_layer_norm=False,
                    )
                    mode_str = "plain"
                self.qc_inside.q_proj = spectral_norm(self.qc_inside.q_proj)
                self.qc_inside.q_out = spectral_norm(self.qc_inside.q_out)

            placement_str = "direct g(Z)" if qi_direct else "post-backbone g(h(Z))"
            qi_total_params = sum(p.numel() for p in self.qc_inside.parameters())
            print(f"    [Quantum-inside] {mode_str}, {placement_str}, "
                  f"{qi_n_qubits} qubits, alpha={self.qi_alpha}, "
                  f"params={qi_total_params}, spectral_norm=on")
        else:
            self.qc_inside = None

        self.deq_solver = None
        if solver == 'torchdeq':
            if TORCHDEQ_AVAILABLE:
                self.deq_solver = get_deq(
                    f_solver='anderson',
                    f_max_iter=max_iter,
                    f_tol=tol,
                    f_stop_mode='rel',
                    b_solver='anderson',
                    b_max_iter=max(max_iter // 2, 20),
                    b_tol=tol * 10,
                    b_stop_mode='rel',
                    ift=True,
                )
            else:
                print("  TorchDEQ not available, falling back to simple solver")
                self.solver = 'simple'

    def _h_only_step(self, Z, W_proj, B, Q, adj, mask):
        """Compute h(Z) = tanh(propagation) without the quantum residual (for L_g estimation)."""
        batch_size = Z.shape[0]
        Z_T = Z.transpose(1, 2)
        WZ = torch.bmm(W_proj.unsqueeze(0).expand(batch_size, -1, -1), Z_T)
        WZA = torch.bmm(WZ, adj.transpose(1, 2))
        out = torch.tanh(WZA + B + Q + self.bias.view(1, -1, 1))
        Z_new = out.transpose(1, 2) * mask.unsqueeze(-1).float()
        return Z_new

    def _phi_step(self, Z, W_proj, B, Q, adj, mask, X=None, q_global=None,
                  topo_node_features=None, topo_graph_features=None):
        batch_size = Z.shape[0]

        if self.use_dynamic_film and X is not None and q_global is not None:
            XZ = torch.cat([X, Z], dim=-1)
            gamma = self.film_scale(XZ) * 1.0 + 0.5
            beta = self.film_shift(XZ) * 0.5
            Q_dynamic = gamma * q_global + beta
            Q_dynamic = Q_dynamic * mask.unsqueeze(-1).float()
            Q = Q_dynamic.transpose(1, 2)

        Z_T = Z.transpose(1, 2)
        WZ = torch.bmm(W_proj.unsqueeze(0).expand(batch_size, -1, -1), Z_T)
        WZA = torch.bmm(WZ, adj.transpose(1, 2))
        out = torch.tanh(WZA + B + Q + self.bias.view(1, -1, 1))
        Z_new = out.transpose(1, 2)
        Z_new = Z_new * mask.unsqueeze(-1).float()

        if self.quantum_inside and self.qc_inside is not None:
            # Paper: SD uses g_ξ(Z); BD uses g_ξ(h_θ(Z)). After the tanh above,
            # Z_new is the backbone output at this iterate; qi_direct picks Z vs Z_new.
            bs, max_n, h = Z_new.shape
            residual_input = Z if self.qi_direct else Z_new
            z_flat = residual_input.reshape(bs * max_n, h)
            if self.qi_topo and topo_node_features is not None and not self.qi_classical:
                tn = topo_node_features.unsqueeze(1).expand(-1, max_n, -1, -1).reshape(
                    bs * max_n, topo_node_features.shape[1], topo_node_features.shape[2])
                tg = topo_graph_features.unsqueeze(1).expand(-1, max_n, -1).reshape(
                    bs * max_n, topo_graph_features.shape[1])
                q_out = self.qc_inside(z_flat, topo_node_features=tn, topo_graph_features=tg)
            else:
                q_out = self.qc_inside(z_flat)
            Z_new = Z_new + self.qi_alpha * q_out.reshape(bs, max_n, h)
            Z_new = Z_new * mask.unsqueeze(-1).float()

        return Z_new

    def _scaled_quantum_residual(
        self,
        Z: torch.Tensor,
        W_proj: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
        X: torch.Tensor = None,
        q_global: torch.Tensor = None,
        topo_node_features: torch.Tensor = None,
        topo_graph_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """α·g(·) at iterate Z, using the same residual placement as _phi_step.

        With ``--no_quantum``, the tanh-input broadcast ``Q`` is zero, but the
        in-loop residual still runs; Q-stats should describe this tensor, not
        the external injection channel.
        """
        if not (self.quantum_inside and self.qc_inside is not None):
            return torch.zeros_like(Z)
        batch_size, max_n, h = Z.shape
        if self.use_dynamic_film and X is not None and q_global is not None:
            XZ = torch.cat([X, Z], dim=-1)
            gamma = self.film_scale(XZ) * 1.0 + 0.5
            beta = self.film_shift(XZ) * 0.5
            Q_dynamic = gamma * q_global + beta
            Q_dynamic = Q_dynamic * mask.unsqueeze(-1).float()
            Q = Q_dynamic.transpose(1, 2)
        Z_T = Z.transpose(1, 2)
        WZ = torch.bmm(W_proj.unsqueeze(0).expand(batch_size, -1, -1), Z_T)
        WZA = torch.bmm(WZ, adj.transpose(1, 2))
        out = torch.tanh(WZA + B + Q + self.bias.view(1, -1, 1))
        Z_new = out.transpose(1, 2) * mask.unsqueeze(-1).float()
        residual_input = Z if self.qi_direct else Z_new
        z_flat = residual_input.reshape(batch_size * max_n, h)
        if self.qi_topo and topo_node_features is not None and not self.qi_classical:
            tn = topo_node_features.unsqueeze(1).expand(-1, max_n, -1, -1).reshape(
                batch_size * max_n, topo_node_features.shape[1], topo_node_features.shape[2])
            tg = topo_graph_features.unsqueeze(1).expand(-1, max_n, -1).reshape(
                batch_size * max_n, topo_graph_features.shape[1])
            q_out = self.qc_inside(z_flat, topo_node_features=tn, topo_graph_features=tg)
        else:
            q_out = self.qc_inside(z_flat)
        scaled = (self.qi_alpha * q_out.reshape(batch_size, max_n, h)) * mask.unsqueeze(-1).float()
        return scaled

    def _compute_wza(self, Z: torch.Tensor, W_proj: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Compute WZA = (W @ Z^T) @ A^T for tanh input diagnostics."""
        batch_size = Z.shape[0]
        Z_T = Z.transpose(1, 2)
        WZ = torch.bmm(W_proj.unsqueeze(0).expand(batch_size, -1, -1), Z_T)
        return torch.bmm(WZ, adj.transpose(1, 2))

    def forward(
        self,
        injection: torch.Tensor,
        adj: torch.Tensor,
        q_features: torch.Tensor,
        mask: torch.Tensor,
        compute_jac_reg: bool = False,
        compute_L_g: bool = False,
        compute_Q_stats: bool = False,
        q_global: torch.Tensor = None,
        topo_node_features: torch.Tensor = None,
        topo_graph_features: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Return fixed point ``z_out`` and solver diagnostics.

        ``injection`` is the encoder output *H* (per-node, padded). ``adj`` is the
        normalized propagation matrix *Ã*. ``q_features`` is the additive *Q* term in
        the tanh pre-activation (for independent injection, *Q* is fixed for the
        whole solve). State- and backbone-dependent *g_ξ* runs inside :meth:`_phi_step` when
        ``quantum_inside`` is enabled.
        """
        batch_size, max_nodes, hidden = injection.shape
        device = injection.device

        kappa_target = self._get_kappa_target(adj)
        self._project_spectral_norm(kappa_target)
        W_proj = self.W

        B = torch.bmm(
            self.Omega.unsqueeze(0).expand(batch_size, -1, -1),
            injection.transpose(1, 2))
        if self.ignn_injection:
            B_pre_norm = B.norm().item() if compute_Q_stats else None
            B = torch.bmm(B, adj)  # IGNN-style diffused injection: B = Omega @ injection @ A
            B_post_norm = B.norm().item() if compute_Q_stats else None
        else:
            B_pre_norm = B_post_norm = None

        if self.use_dynamic_film and q_global is not None:
            Q_static = torch.zeros(batch_size, hidden, max_nodes, device=device)
            q_global_expanded = q_global.expand(-1, max_nodes, -1)
            X_for_film = injection
        else:
            Q_static = q_features.transpose(1, 2)
            q_global_expanded = None
            X_for_film = None

        z0 = torch.zeros(batch_size, max_nodes, hidden, device=device)

        def batched_phi(Z):
            return self._phi_step(
                Z, W_proj, B, Q_static, adj, mask, X_for_film, q_global_expanded,
                topo_node_features=topo_node_features,
                topo_graph_features=topo_graph_features)

        if self.solver == 'torchdeq' and self.deq_solver is not None:
            result = self.deq_solver(batched_phi, z0)
            if isinstance(result, tuple) and len(result) == 2:
                z_traj, info = result
            else:
                z_traj, info = result, {}
            z_out = z_traj[-1] if isinstance(z_traj, (list, tuple)) else z_traj

            n_iter = info.get('nstep', self.max_iter) if isinstance(info, dict) else self.max_iter
            if isinstance(n_iter, torch.Tensor):
                n_iter = n_iter.max().item()
            with torch.no_grad():
                residual = (batched_phi(z_out) - z_out).abs().max().item()
            diagnostics = {
                'n_iter': int(n_iter), 'converged': residual < self.tol,
                'residual': residual, 'solver': 'torchdeq',
            }
        elif self.solver == 'unroll':
            # Full BPTT through `max_iter` Picard iterations. Used by the
            # barren-plateau diagnostic so that gradients flow through every
            # solver step (and, for SD/BD, through every PQC evaluation).
            # Does NOT use the IFT surrogate; memory scales with max_iter.
            Z = z0
            for i in range(self.max_iter):
                Z_new = batched_phi(Z)
                Z = self.damping * Z_new + (1 - self.damping) * Z
            z_out = Z
            with torch.no_grad():
                residual = (batched_phi(z_out) - z_out).abs().max().item()
            diagnostics = {
                'n_iter': int(self.max_iter), 'converged': residual < self.tol,
                'residual': residual, 'solver': 'unroll',
            }
        else:
            with torch.no_grad():
                Z = z0
                res = float('inf')
                for i in range(self.max_iter):
                    Z_new = batched_phi(Z)
                    Z_next = self.damping * Z_new + (1 - self.damping) * Z
                    res = (Z_next - Z).abs().max().item()
                    Z = Z_next
                    if res < self.tol:
                        break
                z_star = Z
            z_star = z_star.detach().requires_grad_(self.training)
            z_out = batched_phi(z_star)
            if self.training:
                z_out = z_star + (z_out - z_star.detach())
            diagnostics = {
                'n_iter': i + 1, 'converged': res < self.tol,
                'residual': res, 'solver': 'simple',
            }

        if compute_jac_reg and self.training:
            diagnostics['jac_reg'] = self._compute_jacobian_reg(z_out, batched_phi, mask)
        else:
            diagnostics['jac_reg'] = torch.tensor(0.0, device=device)

        if compute_L_g and self.quantum_inside:
            with torch.enable_grad():
                L_g_est = self._estimate_L_g(
                    z_out.detach(), W_proj, B, Q_static, adj, mask,
                    topo_node_features=topo_node_features,
                    topo_graph_features=topo_graph_features,
                    n_power_iter=1000,
                )
            diagnostics['L_g'] = L_g_est

        if compute_Q_stats:
            with torch.no_grad():
                # External tanh injection (often zero with --no_quantum); for
                # quantum_inside, report stats of α·g(·) at equilibrium instead.
                if self.quantum_inside and self.qc_inside is not None:
                    Q_q = self._scaled_quantum_residual(
                        z_out, W_proj, B, Q_static, adj, mask,
                        X=X_for_film, q_global=q_global_expanded,
                        topo_node_features=topo_node_features,
                        topo_graph_features=topo_graph_features,
                    )
                    Q = Q_q.transpose(1, 2).contiguous()
                else:
                    Q = Q_static
                B_t = B
                WZA = self._compute_wza(z_out, W_proj, adj)
                # Mask valid nodes: [bs, hidden, max_n], mask [bs, max_n]
                mask3 = mask.unsqueeze(1).expand_as(Q)
                q_flat = Q[mask3]
                b_flat = B_t[mask3]
                wza_flat = WZA[mask3]

                n_valid = q_flat.numel()
                if n_valid == 0:
                    q_mean = q_std = q_abs_mean = q_max_abs = 0.0
                    b_mean = b_std = b_abs_mean = b_max_abs = 0.0
                    wza_abs_mean = 0.0
                else:
                    q_mean = q_flat.mean().item()
                    q_std = q_flat.std().item()
                    q_abs_mean = q_flat.abs().mean().item()
                    q_max_abs = q_flat.abs().max().item()
                    b_mean = b_flat.mean().item()
                    b_std = b_flat.std().item()
                    b_abs_mean = b_flat.abs().mean().item()
                    b_max_abs = b_flat.abs().max().item()
                    wza_abs_mean = wza_flat.abs().mean().item()
                q_norm = Q.norm().item()
                b_norm = B_t.norm().item()
                wza_norm = WZA.norm().item()

                eps = 1e-8
                q_b_ratio_norm = q_norm / (b_norm + eps)
                q_wza_ratio_norm = q_norm / (wza_norm + eps)
                q_b_ratio_abs = q_abs_mean / (b_abs_mean + eps)
                q_wza_ratio_abs = q_abs_mean / (wza_abs_mean + eps)

                diagnostics['Q_mean'] = q_mean
                diagnostics['Q_std'] = q_std
                diagnostics['Q_abs_mean'] = q_abs_mean
                diagnostics['Q_max_abs'] = q_max_abs
                diagnostics['Q_norm'] = q_norm
                diagnostics['B_mean'] = b_mean
                diagnostics['B_std'] = b_std
                diagnostics['B_abs_mean'] = b_abs_mean
                diagnostics['B_max_abs'] = b_max_abs
                diagnostics['B_norm'] = b_norm
                diagnostics['WZA_abs_mean'] = wza_abs_mean
                diagnostics['WZA_norm'] = wza_norm
                diagnostics['Q_B_ratio_norm'] = q_b_ratio_norm
                diagnostics['Q_WZA_ratio_norm'] = q_wza_ratio_norm
                diagnostics['Q_B_ratio_abs'] = q_b_ratio_abs
                diagnostics['Q_WZA_ratio_abs'] = q_wza_ratio_abs
                if self.ignn_injection and B_pre_norm is not None and B_post_norm is not None:
                    diagnostics['B_pre_norm'] = B_pre_norm
                    diagnostics['B_post_norm'] = B_post_norm

        return z_out, diagnostics

    def _compute_jacobian_reg(self, z_star, phi_fn, mask, n_samples=1):
        """Hutchinson's trace estimator for Jacobian Frobenius norm."""
        batch_size, max_nodes, hidden = z_star.shape
        device = z_star.device
        jac_norm_sq = 0.0

        for _ in range(n_samples):
            v = torch.randint(0, 2, z_star.shape, device=device).float() * 2 - 1
            v = v * mask.unsqueeze(-1).float()

            z_star_v = z_star.detach().requires_grad_(True)
            phi_z = phi_fn(z_star_v)
            Jv = torch.autograd.grad(phi_z, z_star_v, v,
                                     create_graph=True, retain_graph=True)[0]
            jac_norm_sq = jac_norm_sq + (Jv ** 2).sum() / (mask.sum() * hidden)

        return jac_norm_sq / n_samples

    def _estimate_L_g(
        self,
        z_out: torch.Tensor,
        W_proj: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
        topo_node_features: Optional[torch.Tensor] = None,
        topo_graph_features: Optional[torch.Tensor] = None,
        n_power_iter: int = 1000,
        n_sample_nodes: int = 32,
    ) -> float:
        """
        Estimate an empirical Jacobian spectral-norm proxy for the quantum residual g
        via power iteration.

        Current implementation:
            g(z) = alpha * qc_inside(z)

        In the reported settings, this empirical estimate remains below 1.
        """
        if not self.quantum_inside or self.qc_inside is None:
            return 0.0
        batch_size, max_nodes, hidden = z_out.shape
        device = z_out.device
        with torch.no_grad():
            h_z = self._h_only_step(z_out, W_proj, B, Q, adj, mask)
        z_flat = h_z.reshape(batch_size * max_nodes, hidden)
        mask_flat = mask.reshape(batch_size * max_nodes, 1)
        n_valid = int(mask_flat.sum().item())
        if n_valid == 0:
            return 0.0
        valid_idx = mask_flat.squeeze(-1).nonzero(as_tuple=True)[0]
        if len(valid_idx) > n_sample_nodes:
            perm = torch.randperm(len(valid_idx), device=device)[:n_sample_nodes]
            valid_idx = valid_idx[perm]
        z_sample = z_flat[valid_idx].detach()
        tn_sample = topo_node_features[valid_idx] if topo_node_features is not None else None
        tg_sample = topo_graph_features[valid_idx] if topo_graph_features is not None else None

        def g_fn(z):
            if self.qi_topo and tn_sample is not None and tg_sample is not None:
                qc_out = self.qc_inside(z, topo_node_features=tn_sample, topo_graph_features=tg_sample)
            else:
                qc_out = self.qc_inside(z)
            return self.qi_alpha * qc_out

        try:
            from torch.autograd.functional import jvp
        except ImportError:
            return float('nan')

        v = torch.randn_like(z_sample, device=device)
        v = v / (v.norm() + 1e-10)
        sigma = 0.0
        for _ in range(n_power_iter):
            z_in = z_sample.requires_grad_(True)
            _, u = jvp(g_fn, z_in, v)
            u = u.detach()
            sigma = (u ** 2).sum().sqrt().item()
            if sigma < 1e-10:
                break
            v = u / sigma
            z_in = z_in.detach().requires_grad_(True)
            (g_fn(z_in) * v).sum().backward()
            w = z_in.grad.detach()
            w = w / (w.norm() + 1e-10)
            v = w
        return float(sigma)

    def _project_spectral_norm(self, kappa_target: float):
        """Project W in-place onto ||W||_2 <= kappa_target via spectral clipping.

        Uses power iteration to estimate sigma_max(W), then scales if needed.
        This is the ℓ₂ analog of IGNN's ℓ_∞ row-wise projection.
        """
        with torch.no_grad():
            W = self.W
            u = self._w_u
            if not self._w_u_initialized:
                u = F.normalize(u, dim=0)
                self._w_u_initialized.fill_(True)

            # 3 steps of power iteration for sigma_max(W)
            for _ in range(3):
                v = F.normalize(W.T @ u, dim=0)
                u = F.normalize(W @ v, dim=0)
            sigma = u @ W @ v

            self._w_u.copy_(u)

            if sigma > kappa_target:
                self.W.data.mul_(kappa_target / sigma)

    def _estimate_A_spectral_radius(self, adj: torch.Tensor, n_iters: int = 5) -> float:
        """Estimate ||A||_2 via power iteration on batched adjacency.

        For symmetric A: ||A||_2 = spectral radius = largest eigenvalue.
        """
        with torch.no_grad():
            bs, n, _ = adj.shape
            v = torch.randn(bs, n, 1, device=adj.device)
            v = F.normalize(v, dim=1)
            for _ in range(n_iters):
                Av = torch.bmm(adj, v)
                v = F.normalize(Av, dim=1)
            # Rayleigh quotient: v^T A v
            rho = (v.transpose(1, 2) @ torch.bmm(adj, v)).squeeze()
            return rho.abs().mean().item()

    def _get_kappa_target(self, adj: torch.Tensor) -> float:
        """Compute kappa target: ||W||_2 <= kappa / ||A||_2."""
        A_spectral = self._estimate_A_spectral_radius(adj)
        A_spectral = max(A_spectral, 1e-6)
        return self.kappa / A_spectral


# =============================================================================
# Decoder
# =============================================================================

class GNNDecoder(nn.Module):
    """GNN decoder for post-implicit feature refinement."""

    def __init__(self, hidden_dim: int, n_layers: int = 2, dropout: float = 0.1,
                 use_layer_norm: bool = False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

    def forward(self, h, adj, mask=None):
        """
        Args:
            h: [batch, max_nodes, hidden]
            adj: [batch, max_nodes, max_nodes]
            mask: [batch, max_nodes]
        """
        batch_size, max_nodes, hidden_dim = h.shape
        for layer, norm in zip(self.layers, self.norms):
            neighbor_h = torch.bmm(adj, h)
            h_flat = (h + neighbor_h).reshape(-1, hidden_dim)
            h_new = F.dropout(F.relu(norm(layer(h_flat))),
                              p=self.dropout, training=self.training)
            h_new = h_new.reshape(batch_size, max_nodes, hidden_dim)
            h = h + h_new
            if mask is not None:
                h = h * mask.unsqueeze(-1).float()
        return h


# =============================================================================
# Full QIGNN Model
# =============================================================================

class TopoAwareQIGNN(nn.Module):
    r"""
    End-to-end model for the paper: encoder :math:`H`, optional encoder-side PQC,
    then (optional) :class:`BatchedImplicitCore` for :math:`Z^\star`, readout, classifier.

    **Pathways (paper, *The Three Injection Pathways*):**
    * ``q_ind_node``: independent injection :math:`Q_{\mathrm{IN}}` from per-node
      encoder features and topology (``quantum_node``) → passed as static ``q_features``
      into the implicit map (iterate-independent conditioning).
    * ``quantum_inside`` + ``qi_direct`` / not: in-loop residual
      :math:`\alpha g_\xi` on :math:`Z` vs. on backbone output, shared ``qc_inside`` module.

    The ``--no_quantum`` training flag turns off the *separate* encoder graph-level
    PQC; reported IN/SD/BD runs use it so only the pathway above carries the quantum
    contribution described in the paper.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        n_qubits: int = 4,
        circuit_reps: int = 3,
        n_encoder_layers: int = 5,
        jk_mode: str = 'sum',
        dropout: float = 0.1,
        max_cycle_length: int = 20,
        use_topo_encoding: bool = True,
        use_topo_ising: bool = True,
        use_competitive_gate: bool = True,
        implicit_global: bool = False,
        implicit_self_loops: bool = False,
        normalize_implicit_adj: bool = True,
        kappa: float = 0.999,
        solver: str = 'torchdeq',
        max_iter: int = 50,
        tol: float = 1e-6,
        use_decoder: bool = True,
        n_decoder_layers: int = 2,
        pooling: str = 'sum',
        use_quantum: bool = True,
        use_film: bool = False,
        dynamic_film: bool = False,
        simple_encoder: bool = False,
        min_encoder: bool = False,
        # Local Quantum Aggregator (LQA) options
        lqa: bool = False,
        lqa_max_neighbors: int = 4,
        lqa_qubits_per_neighbor: int = 4,
        lqa_conv_layers: int = 2,
        # Stochastic topology conditioning
        topo_drop_enc: float = 0.0,
        topo_drop_ising: float = 0.0,
        topo_drop_gate: float = 0.0,
        jac_reg: float = 0.0,
        q_inj_scale: float = 1.0,
        use_layer_norm: bool = False,
        quantum_inside: bool = False,
        qi_n_qubits: int = 4,
        qi_circuit_reps: int = 1,
        qi_alpha: float = 0.1,
        qi_topo: bool = False,
        perm_invariant: bool = False,
        no_q_inject: bool = False,
        q_inj_node_cond: bool = False,
        ignn_injection: bool = False,
        qi_direct: bool = False,
        qi_classical: bool = False,
        q_ind_node: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.no_q_inject = no_q_inject
        self.ignn_injection = ignn_injection
        self.q_inj_node_cond = q_inj_node_cond
        self.q_ind_node = q_ind_node
        self.n_qubits = n_qubits
        self.implicit_global = implicit_global
        self.implicit_self_loops = implicit_self_loops
        self.normalize_implicit_adj = normalize_implicit_adj
        self.use_decoder = use_decoder
        self.use_quantum = use_quantum
        self.use_film = use_film
        self.dynamic_film = dynamic_film
        self.pooling_type = pooling
        self.jac_reg = jac_reg
        self.q_inj_scale = q_inj_scale
        self.topo_drop_enc = topo_drop_enc
        self.topo_drop_ising = topo_drop_ising
        self.topo_drop_gate = topo_drop_gate

        # --- Encoder ---
        if lqa:
            from .lqa import LocalQuantumGINEncoder
            self.encoder = LocalQuantumGINEncoder(
                in_features=in_features, hidden_dim=hidden_dim,
                n_layers=n_encoder_layers,
                n_qubits_per_neighbor=lqa_qubits_per_neighbor,
                max_neighbors=lqa_max_neighbors,
                conv_layers=lqa_conv_layers,
                dropout=dropout, use_layer_norm=use_layer_norm,
                perm_invariant=perm_invariant)
        elif min_encoder:
            self.encoder = MinEncoder(in_features, hidden_dim, use_relu=False)
        elif simple_encoder:
            self.encoder = SimpleEncoder(
                in_features, hidden_dim, n_layers=1,
                use_layer_norm=use_layer_norm)
        else:
            self.encoder = GINEncoder(
                in_features, hidden_dim, n_encoder_layers, dropout,
                jk_mode=jk_mode, use_layer_norm=use_layer_norm)

        # --- Graph pooling ---
        self.graph_pool = BatchedGraphPooling(hidden_dim, pooling=pooling)

        # --- Topology dimensions ---
        self.topo_node_dim = max_cycle_length - 2 + 3
        self.topo_graph_dim = 4 + (max_cycle_length - 2)

        # --- Quantum layer ---
        if use_quantum:
            self.node_to_qubit = nn.Sequential(
                nn.Linear(self.topo_node_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_qubits),
            )
            edge_pairs = get_edge_pairs(n_qubits, 'linear')
            self.quantum = TopoAwareQuantumLayer(
                hidden_dim=hidden_dim, n_qubits=n_qubits,
                edge_pairs=edge_pairs, circuit_reps=circuit_reps,
                topo_node_dim=self.topo_node_dim,
                topo_graph_dim=self.topo_graph_dim,
                use_topo_encoding=use_topo_encoding,
                use_topo_ising=use_topo_ising,
                use_competitive_gate=use_competitive_gate,
                perm_invariant=perm_invariant,
                use_layer_norm=True)
            self.q_node_proj = nn.Linear(hidden_dim, n_qubits)

            if dynamic_film:
                self.film_scale = None
                self.film_shift = None
            elif use_film:
                self.film_scale = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
                self.film_shift = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            else:
                self.film_scale = None
                self.film_shift = None
            if q_inj_node_cond:
                self.q_inj_gate = nn.Linear(hidden_dim * 2, hidden_dim)
            else:
                self.q_inj_gate = None
        else:
            self.node_to_qubit = None
            self.quantum = None
            self.q_node_proj = None
            self.film_scale = None
            self.film_shift = None
            self.q_inj_gate = None

        # --- Independent node-level quantum injection ---
        if q_ind_node:
            edge_pairs_ind = get_edge_pairs(n_qubits, 'linear')
            self.quantum_node = TopoAwareQuantumLayer(
                hidden_dim=hidden_dim, n_qubits=n_qubits,
                edge_pairs=edge_pairs_ind, circuit_reps=circuit_reps,
                topo_node_dim=self.topo_node_dim,
                topo_graph_dim=self.topo_graph_dim,
                use_topo_encoding=use_topo_encoding,
                use_topo_ising=use_topo_ising,
                use_competitive_gate=use_competitive_gate,
                perm_invariant=perm_invariant,
                use_layer_norm=True)
            qi_node_params = sum(p.numel() for p in self.quantum_node.parameters())
            print(f"    [Q-ind-node] per-node quantum injection, "
                  f"{n_qubits} qubits, reps={circuit_reps}, "
                  f"params={qi_node_params}")
        else:
            self.quantum_node = None

        # --- Implicit core ---
        if implicit_global:
            topo_node_dim = max_cycle_length - 2 + 3
            topo_graph_dim = 4 + (max_cycle_length - 2)
            self.implicit_core = BatchedImplicitCore(
                hidden_dim=hidden_dim, kappa=kappa, max_iter=max_iter,
                tol=tol, solver=solver, use_dynamic_film=dynamic_film,
                quantum_inside=quantum_inside,
                qi_n_qubits=qi_n_qubits,
                qi_circuit_reps=qi_circuit_reps,
                qi_alpha=qi_alpha,
                qi_topo=qi_topo,
                qi_topo_node_dim=topo_node_dim,
                qi_topo_graph_dim=topo_graph_dim,
                perm_invariant=perm_invariant,
                ignn_injection=ignn_injection,
                qi_direct=qi_direct,
                qi_classical=qi_classical)
        else:
            self.implicit_core = None

        # --- Decoder ---
        if implicit_global and use_decoder:
            self.decoder = GNNDecoder(
                hidden_dim, n_decoder_layers, dropout,
                use_layer_norm=use_layer_norm)
        else:
            self.decoder = None

        # --- Classifier ---
        # JK heads only for pure GIN baseline (no quantum, no implicit, GINEncoder)
        self._use_jk = (not implicit_global and not use_quantum
                        and isinstance(self.encoder, GINEncoder))
        if implicit_global:
            classifier_in = hidden_dim
        elif use_quantum:
            classifier_in = hidden_dim * 2  # concat GNN + quantum features
        elif self._use_jk:
            classifier_in = None  # JK heads handle classification directly
        else:
            classifier_in = hidden_dim

        if classifier_in is not None:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_in, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
            self.jk_heads = None
        else:
            self.jk_heads = nn.ModuleList()
            for layer in range(n_encoder_layers + 1):
                in_d = in_features if layer == 0 else hidden_dim
                self.jk_heads.append(nn.Linear(in_d, num_classes))
            self.classifier = None

        self.dropout = dropout
        self.num_classes = num_classes

    def forward(
        self,
        data: Data,
        topo_features: Optional[Dict[str, torch.Tensor]] = None,
        compute_L_g: bool = False,
        compute_Q_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        device = x.device
        diagnostics = {}

        # 1. Encode
        h = self.encoder(x, edge_index, batch)
        if hasattr(self.encoder, '_last_solve_info') and self.encoder._last_solve_info is not None:
            diagnostics['local'] = self.encoder._last_solve_info

        # 2. Pool
        h_graph = self.graph_pool(h, batch)

        # 3. Quantum
        if self.use_quantum:
            if topo_features is not None:
                node_topo = topo_features['combined_node_features'].to(device)
                graph_topo = topo_features['graph_cycle_features'].to(device)
                node_topo_pooled = global_mean_pool(node_topo, batch)
                node_topo_qubits = node_topo_pooled.unsqueeze(1).expand(-1, self.n_qubits, -1)
            else:
                node_topo_qubits = None
                graph_topo = None

            if self.training:
                enable_enc = torch.rand(1).item() > self.topo_drop_enc
                enable_ising = torch.rand(1).item() > self.topo_drop_ising
                enable_gate = torch.rand(1).item() > self.topo_drop_gate
            else:
                enable_enc = enable_ising = enable_gate = True

            h_quantum = self.quantum(
                h_graph, node_topo_qubits, graph_topo,
                enable_encoding=enable_enc, enable_ising=enable_ising,
                enable_gate=enable_gate)
        else:
            h_quantum = torch.zeros_like(h_graph)

        if self.implicit_global:
            padded_h, padded_adj, node_mask, bs, max_n = pyg_to_batched_dense(
                data, h, add_self_loops=self.implicit_self_loops,
                normalize_adj=self.normalize_implicit_adj)

            q_global_for_dynamic = None
            if self.q_ind_node and self.quantum_node is not None:
                # IN (paper): Q_IN(H, τ(A)) row-wise, fixed before the fixed-point solve,
                # added inside σ with H Ω^T (not an outer h_θ + Q sum).
                # quantum_node: encode--unitary--measure from (h_v, node/topology descriptors)
                if topo_features is not None:
                    ind_node_topo = topo_features['combined_node_features'].to(device)
                    ind_graph_topo = topo_features['graph_cycle_features'].to(device)
                    ind_node_topo_q = ind_node_topo.unsqueeze(1).expand(
                        -1, self.n_qubits, -1)
                    ind_graph_topo_pn = ind_graph_topo[batch]
                else:
                    ind_node_topo_q = None
                    ind_graph_topo_pn = None

                if self.training:
                    en_enc = torch.rand(1).item() > self.topo_drop_enc
                    en_ising = torch.rand(1).item() > self.topo_drop_ising
                    en_gate = torch.rand(1).item() > self.topo_drop_gate
                else:
                    en_enc = en_ising = en_gate = True

                q_per_node = self.quantum_node(
                    h, ind_node_topo_q, ind_graph_topo_pn,
                    enable_encoding=en_enc, enable_ising=en_ising,
                    enable_gate=en_gate)

                # Pad PyG flat → dense [bs, max_n, hidden]
                q_node_features = torch.zeros(bs, max_n, self.hidden_dim, device=device)
                _npg = torch.bincount(batch, minlength=bs)
                _off = torch.cat([torch.tensor([0], device=device),
                                  _npg.cumsum(0)[:-1]])
                for b_idx in range(bs):
                    n = _npg[b_idx].item()
                    o = _off[b_idx].item()
                    q_node_features[b_idx, :n] = q_per_node[o:o + n]
                q_node_features = self.q_inj_scale * q_node_features
                q_node_features = q_node_features * node_mask.unsqueeze(-1).float()
            elif self.use_quantum and not self.no_q_inject:
                q_global = self.q_inj_scale * h_quantum.unsqueeze(1)
                if self.dynamic_film:
                    q_global_for_dynamic = q_global
                    q_node_features = torch.zeros(bs, max_n, self.hidden_dim, device=device)
                elif self.use_film:
                    gamma = self.film_scale(padded_h) * 1.0 + 0.5
                    beta = self.film_shift(padded_h) * 0.5
                    q_node_features = gamma * q_global + beta
                elif self.q_inj_node_cond and self.q_inj_gate is not None:
                    qg = q_global.expand(-1, max_n, -1)
                    gate = torch.sigmoid(self.q_inj_gate(torch.cat([padded_h, qg], dim=-1)))
                    q_node_features = gate * qg
                else:
                    q_node_features = q_global.expand(-1, max_n, -1)
                q_node_features = q_node_features * node_mask.unsqueeze(-1).float()
            else:
                q_node_features = torch.zeros(bs, max_n, self.hidden_dim, device=device)

            # Prepare topo features for quantum-inside if enabled
            qi_topo_node = None
            qi_topo_graph = None
            if (self.implicit_core.quantum_inside and self.implicit_core.qi_topo
                    and topo_features is not None):
                node_topo_raw = topo_features['combined_node_features'].to(device)
                graph_topo_raw = topo_features['graph_cycle_features'].to(device)
                n_qubits_qi = self.implicit_core.qc_inside.n_qubits
                node_topo_pooled_qi = global_mean_pool(node_topo_raw, batch)
                qi_topo_node = node_topo_pooled_qi.unsqueeze(1).expand(
                    -1, n_qubits_qi, -1)
                qi_topo_graph = graph_topo_raw

            compute_jac = self.training and self.jac_reg > 0
            z_star, global_diag = self.implicit_core(
                padded_h, padded_adj, q_node_features, node_mask,
                compute_jac_reg=compute_jac, compute_L_g=compute_L_g,
                compute_Q_stats=compute_Q_stats,
                q_global=q_global_for_dynamic,
                topo_node_features=qi_topo_node,
                topo_graph_features=qi_topo_graph)
            diagnostics['global'] = global_diag

            h_final = z_star
            if self.decoder is not None:
                h_final = self.decoder(h_final, padded_adj, node_mask)

            h_final_masked = h_final * node_mask.unsqueeze(-1).float()
            # Use graph_pool for readout (respects --pooling: sum/mean/max/concat/attention)
            batch_idx = node_mask.nonzero(as_tuple=True)[0]
            h_flat = h_final_masked[node_mask]
            h_graph_final = self.graph_pool(h_flat, batch_idx)
            out = self.classifier(h_graph_final)
        else:
            if self.use_quantum:
                out = self.classifier(torch.cat([h_graph, h_quantum], dim=-1))
            elif self._use_jk:
                hidden_rep = self.encoder.get_all_layers(x, edge_index, batch)
                score = torch.zeros(h_graph.shape[0], self.num_classes, device=device)
                for layer, h_layer in enumerate(hidden_rep):
                    pooled = global_add_pool(h_layer, batch)
                    pred = F.dropout(self.jk_heads[layer](pooled),
                                     self.dropout, training=self.training)
                    score = score + pred
                out = score
            else:
                out = self.classifier(h_graph)

        return out, diagnostics
