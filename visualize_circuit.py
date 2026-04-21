"""
Visualize QIGNN quantum circuits and graph topology for paper figures.

Generates:
1. Quantum circuit diagrams showing the full LQA pipeline
2. Example graph topology with neighbor sampling
3. Gate depth statistics

Usage:
    python visualize_circuit.py --max_neighbors 4 --n_qubits_per_neighbor 2 --conv_layers 1
    python visualize_circuit.py --max_neighbors 10 --n_qubits_per_neighbor 1 --conv_layers 1
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import os


def compute_gate_depth(max_neighbors, n_qubits_per_neighbor, conv_layers):
    """Compute gate depth statistics for the quantum circuit."""
    n = max_neighbors * n_qubits_per_neighbor
    
    stats = {
        'total_qubits': n,
        'max_neighbors': max_neighbors,
        'n_qubits_per_neighbor': n_qubits_per_neighbor,
        'conv_layers': conv_layers,
        'state_dim': 2 ** n,
    }
    
    # Encoding layer: 3 gates per qubit (RX, RY, RZ)
    encoding_depth = 3  # RX, RY, RZ applied sequentially per qubit
    encoding_gates = n * 3  # Total single-qubit gates
    
    # Per conv layer
    conv_depths = []
    current_qubits = n
    total_zz_gates = 0
    total_rot_gates = 0
    
    for layer in range(conv_layers):
        n_zz = current_qubits - 1  # ZZ between adjacent pairs
        n_rot = current_qubits * 3  # RX, RY, RZ per qubit
        
        # ZZ gates can be parallelized (even/odd pairs), depth = ceil(n_zz / floor(n/2))
        # But in linear chain, alternating even/odd gives depth 2
        zz_depth = 2 if current_qubits > 2 else 1
        rot_depth = 3  # RX, RY, RZ sequential
        
        conv_depths.append({
            'qubits': current_qubits,
            'zz_gates': n_zz,
            'rot_gates': n_rot,
            'zz_depth': zz_depth,
            'rot_depth': rot_depth,
            'layer_depth': zz_depth + rot_depth,
        })
        
        total_zz_gates += n_zz
        total_rot_gates += n_rot
        
        # After pooling
        current_qubits = max((current_qubits + 1) // 2, 1)
    
    stats['encoding_depth'] = encoding_depth
    stats['encoding_gates'] = encoding_gates
    stats['conv_layers_detail'] = conv_depths
    stats['total_single_qubit_gates'] = encoding_gates + total_rot_gates
    stats['total_two_qubit_gates'] = total_zz_gates
    stats['total_gates'] = encoding_gates + total_rot_gates + total_zz_gates
    stats['total_depth'] = encoding_depth + sum(d['layer_depth'] for d in conv_depths)
    stats['final_qubits'] = current_qubits
    
    return stats


def draw_quantum_circuit(max_neighbors, n_qubits_per_neighbor, conv_layers, 
                          output_path='circuit_diagram.pdf'):
    """Draw the quantum circuit diagram for the paper."""
    n = max_neighbors * n_qubits_per_neighbor
    
    fig, ax = plt.subplots(1, 1, figsize=(16, max(4, n * 0.6)))
    
    # Colors for different neighbor groups
    colors = plt.cm.Set3(np.linspace(0, 1, max_neighbors))
    
    # Layout parameters
    x_start = 0.5
    x_step = 1.2
    y_positions = np.arange(n)[::-1]  # Top to bottom
    wire_y = {i: y_positions[i] for i in range(n)}
    
    # Draw qubit wires and labels
    x_end = x_start + x_step * (3 + conv_layers * 6 + 3)  # Rough estimate
    for i in range(n):
        neighbor_idx = i // n_qubits_per_neighbor
        qubit_in_group = i % n_qubits_per_neighbor
        color = colors[neighbor_idx]
        
        ax.plot([0, x_end], [wire_y[i], wire_y[i]], '-', color='gray', linewidth=0.5, zorder=0)
        
        # Qubit label
        label = f'$|0\\rangle_{{u_{neighbor_idx},q_{qubit_in_group}}}$'
        ax.text(-0.3, wire_y[i], label, ha='right', va='center', fontsize=9)
        
        # Color bar on the left to show neighbor groups
        rect = FancyBboxPatch((-0.6, wire_y[i] - 0.35), 0.15, 0.7 if n_qubits_per_neighbor > 1 else 0.5,
                               boxstyle="round,pad=0.05", facecolor=color, edgecolor='none', alpha=0.6)
        if qubit_in_group == 0:
            ax.add_patch(rect)
    
    # Helper to draw a gate box
    def draw_gate(x, y, label, color='white', width=0.8, fontsize=8):
        rect = FancyBboxPatch((x - width/2, y - 0.35), width, 0.7,
                               boxstyle="round,pad=0.05", facecolor=color, 
                               edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, weight='bold')
    
    # Helper to draw ZZ gate (two-qubit)
    def draw_zz(x, y1, y2, label='ZZ'):
        ax.plot([x, x], [y1, y2], 'k-', linewidth=2)
        ax.plot(x, y1, 'ko', markersize=8, zorder=5)
        ax.plot(x, y2, 'ko', markersize=8, zorder=5)
        ax.text(x + 0.15, (y1 + y2) / 2, f'$U_{{ZZ}}$', ha='left', va='center', fontsize=7, color='darkblue')
    
    # ============================================================
    # Stage 1: Data Encoding (RX, RY, RZ per qubit)
    # ============================================================
    x = x_start
    ax.text(x + x_step, max(wire_y.values()) + 0.8, 'Data Encoding', 
            ha='center', va='bottom', fontsize=11, weight='bold', color='darkgreen')
    
    for gate_name, gate_color in [('$R_X$', '#FFE0E0'), ('$R_Y$', '#E0FFE0'), ('$R_Z$', '#E0E0FF')]:
        for i in range(n):
            neighbor_idx = i // n_qubits_per_neighbor
            draw_gate(x, wire_y[i], gate_name, color=gate_color, fontsize=8)
        x += x_step
    
    # Separator
    x += 0.3
    ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    x += 0.3
    
    # ============================================================
    # Stage 2: Convolution layers (ZZ + Rotations)
    # ============================================================
    current_qubits = n
    active_qubits = list(range(n))
    
    for layer in range(conv_layers):
        layer_x_start = x
        
        ax.text(x + x_step * 2, max(wire_y[q] for q in active_qubits) + 0.8, 
                f'Conv Layer {layer+1}', 
                ha='center', va='bottom', fontsize=11, weight='bold', color='darkblue')
        
        # ZZ entangling gates (linear chain)
        # Draw even pairs first, then odd pairs
        for parity in [0, 1]:
            for idx in range(parity, len(active_qubits) - 1, 2):
                q1 = active_qubits[idx]
                q2 = active_qubits[idx + 1]
                draw_zz(x, wire_y[q1], wire_y[q2])
            x += x_step
        
        # Rotation gates
        for gate_name, gate_color in [('$R_X$', '#FFE0E0'), ('$R_Y$', '#E0FFE0'), ('$R_Z$', '#E0E0FF')]:
            for q in active_qubits:
                draw_gate(x, wire_y[q], gate_name, color=gate_color, fontsize=8)
            x += x_step
        
        # Separator
        x += 0.3
        
        # Pooling: trace out even-indexed qubits
        pooled_out = [active_qubits[i] for i in range(0, len(active_qubits), 2)]
        active_qubits = [active_qubits[i] for i in range(1, len(active_qubits), 2)]
        
        if pooled_out:
            # Draw pool markers
            for q in pooled_out:
                ax.text(x, wire_y[q], '⊗', ha='center', va='center', fontsize=14, color='red')
                # Dashed line after pool
                ax.plot([x + 0.3, x_end], [wire_y[q], wire_y[q]], '--', 
                        color='lightgray', linewidth=0.5, zorder=0)
            ax.text(x, max(wire_y[q] for q in pooled_out) + 0.8, 'Pool',
                    ha='center', va='bottom', fontsize=10, weight='bold', color='red')
        
        x += 0.8
        current_qubits = len(active_qubits)
    
    # ============================================================
    # Stage 3: Measurement
    # ============================================================
    ax.text(x + 0.3, max(wire_y[q] for q in active_qubits) + 0.8, 'Measure',
            ha='center', va='bottom', fontsize=11, weight='bold', color='purple')
    
    for q in active_qubits:
        # Meter symbol
        draw_gate(x + 0.3, wire_y[q], '$\\langle Z \\rangle$', color='#F0E0FF', fontsize=8)
    
    x_end = x + 1.5
    
    # Update wire lengths
    for i in range(n):
        if i in active_qubits:
            ax.plot([0, x_end], [wire_y[i], wire_y[i]], '-', color='gray', linewidth=0.5, zorder=0)
    
    # Neighbor group labels on the right
    for k in range(max_neighbors):
        qubits_in_group = [k * n_qubits_per_neighbor + j for j in range(n_qubits_per_neighbor)]
        y_center = np.mean([wire_y[q] for q in qubits_in_group])
        ax.text(x_end + 0.3, y_center, f'$u_{k}$', ha='left', va='center', 
                fontsize=10, color=colors[k], weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[k], alpha=0.3))
    
    ax.set_xlim(-1, x_end + 1)
    ax.set_ylim(min(wire_y.values()) - 1, max(wire_y.values()) + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    title = f'QIGNN Local Quantum Aggregator: K={max_neighbors} neighbors, q={n_qubits_per_neighbor} qubits/neighbor, {conv_layers} conv layer(s)'
    ax.set_title(title, fontsize=13, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Circuit diagram saved to {output_path}")
    plt.close()


def draw_graph_topology_example(output_path='graph_topology_example.pdf'):
    """Draw an example molecular graph with neighbor sampling highlighted."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ============================================================
    # Left: Example graph with center node and sampled neighbors
    # ============================================================
    ax = axes[0]
    
    # Create an example molecular-like graph
    G = nx.karate_club_graph()
    # Use a smaller subgraph for clarity
    center = 0
    subgraph_nodes = list(nx.ego_graph(G, center, radius=2).nodes())[:15]
    G_sub = G.subgraph(subgraph_nodes).copy()
    
    pos = nx.spring_layout(G_sub, seed=42, k=1.5)
    
    # Get neighbors of center
    neighbors = list(G_sub.neighbors(center))[:4]  # K=4
    other_nodes = [n for n in G_sub.nodes() if n != center and n not in neighbors]
    
    # Draw edges
    nx.draw_networkx_edges(G_sub, pos, ax=ax, alpha=0.3, width=1)
    
    # Highlight edges from center to sampled neighbors
    center_edges = [(center, n) for n in neighbors]
    nx.draw_networkx_edges(G_sub, pos, edgelist=center_edges, ax=ax, 
                           edge_color='red', width=2.5, alpha=0.8)
    
    # Draw nodes
    nx.draw_networkx_nodes(G_sub, pos, nodelist=other_nodes, ax=ax,
                           node_color='lightgray', node_size=300, edgecolors='gray')
    nx.draw_networkx_nodes(G_sub, pos, nodelist=neighbors, ax=ax,
                           node_color=['#FF9999', '#99FF99', '#9999FF', '#FFFF99'][:len(neighbors)],
                           node_size=500, edgecolors='black', linewidths=2)
    nx.draw_networkx_nodes(G_sub, pos, nodelist=[center], ax=ax,
                           node_color='red', node_size=700, edgecolors='black', linewidths=3)
    
    # Labels
    labels = {center: '$v$'}
    for i, n in enumerate(neighbors):
        labels[n] = f'$u_{i}$'
    nx.draw_networkx_labels(G_sub, pos, labels, ax=ax, font_size=12, font_weight='bold')
    
    ax.set_title('(a) Graph with K=4 sampled neighbors', fontsize=13, pad=10)
    ax.axis('off')
    
    # ============================================================
    # Right: GIN update formula visualization
    # ============================================================
    ax = axes[1]
    ax.axis('off')
    
    # Draw the GIN-quantum update pipeline
    y_top = 0.95
    
    ax.text(0.5, y_top, 'GIN-Quantum Update Rule', ha='center', va='top', 
            fontsize=14, weight='bold', transform=ax.transAxes)
    
    formula_lines = [
        r'$h_v^{(k+1)} = \mathrm{MLP}^{(k)}\!\left((1+\varepsilon^{(k)}) \cdot h_v^{(k)} + \mathcal{Q}^{(k)}\right)$',
        '',
        r'where $\mathcal{Q}^{(k)} = W_{\mathrm{out}} \cdot \langle Z \rangle_{\mathrm{final}}$',
        '',
        r'$\langle Z \rangle$ from quantum circuit:',
        r'  1. Encode: $R_X R_Y R_Z(\theta_u)$ per neighbor',
        r'  2. Entangle: $U_{ZZ}(\eta)$ across neighbors',
        r'  3. Pool: trace out alternating qubits',
        r'  4. Measure: $\langle Z_i \rangle$ on remaining',
    ]
    
    for i, line in enumerate(formula_lines):
        y = y_top - 0.08 - i * 0.08
        fontsize = 13 if i == 0 else 11
        ax.text(0.5, y, line, ha='center', va='top', fontsize=fontsize, 
                transform=ax.transAxes, family='serif' if line.startswith('$') else 'sans-serif')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Graph topology example saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize QIGNN quantum circuits')
    parser.add_argument('--max_neighbors', type=int, default=4)
    parser.add_argument('--n_qubits_per_neighbor', type=int, default=2)
    parser.add_argument('--conv_layers', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='figures')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ============================================================
    # 1. Gate depth statistics
    # ============================================================
    print("=" * 60)
    print("Gate Depth Statistics")
    print("=" * 60)
    
    # Print stats for multiple configs
    configs = [
        (4, 2, 1, "NCI1/PROTEINS"),
        (4, 1, 1, "Small degree"),
        (8, 2, 1, "IMDB-BINARY"),
        (10, 1, 1, "REDDIT/COLLAB"),
    ]
    
    print(f"\n{'Config':<25} {'Qubits':<8} {'1Q gates':<10} {'2Q gates':<10} {'Total':<8} {'Depth':<8} {'State dim':<12}")
    print("-" * 90)
    
    for K, q, L, name in configs:
        stats = compute_gate_depth(K, q, L)
        print(f"K={K}, q={q}, L={L} ({name})"
              f"  {stats['total_qubits']:<8}"
              f"  {stats['total_single_qubit_gates']:<10}"
              f"  {stats['total_two_qubit_gates']:<10}"
              f"  {stats['total_gates']:<8}"
              f"  {stats['total_depth']:<8}"
              f"  {stats['state_dim']:<12}")
    
    # Detailed stats for the requested config
    print(f"\n{'=' * 60}")
    print(f"Detailed: K={args.max_neighbors}, q={args.n_qubits_per_neighbor}, L={args.conv_layers}")
    print(f"{'=' * 60}")
    stats = compute_gate_depth(args.max_neighbors, args.n_qubits_per_neighbor, args.conv_layers)
    print(f"  Total qubits: {stats['total_qubits']}")
    print(f"  State vector dim: {stats['state_dim']}")
    print(f"  Encoding: {stats['encoding_gates']} gates, depth {stats['encoding_depth']}")
    for i, layer in enumerate(stats['conv_layers_detail']):
        print(f"  Conv layer {i+1}: {layer['qubits']} qubits, "
              f"{layer['zz_gates']} ZZ + {layer['rot_gates']} rot gates, "
              f"depth {layer['layer_depth']}")
    print(f"  Final qubits (after pooling): {stats['final_qubits']}")
    print(f"  Total gates: {stats['total_gates']} ({stats['total_single_qubit_gates']} 1Q + {stats['total_two_qubit_gates']} 2Q)")
    print(f"  Total depth: {stats['total_depth']}")
    
    # ============================================================
    # 2. Circuit diagram
    # ============================================================
    print(f"\nGenerating circuit diagram...")
    draw_quantum_circuit(
        args.max_neighbors, args.n_qubits_per_neighbor, args.conv_layers,
        output_path=os.path.join(args.output_dir, 
            f'circuit_K{args.max_neighbors}_q{args.n_qubits_per_neighbor}_L{args.conv_layers}.pdf')
    )
    
    # Also generate for the other common config
    if args.n_qubits_per_neighbor != 1:
        print(f"Generating circuit diagram (q=1 variant)...")
        draw_quantum_circuit(
            args.max_neighbors, 1, args.conv_layers,
            output_path=os.path.join(args.output_dir, 
                f'circuit_K{args.max_neighbors}_q1_L{args.conv_layers}.pdf')
        )
    
    # ============================================================
    # 3. Graph topology example
    # ============================================================
    print(f"\nGenerating graph topology example...")
    draw_graph_topology_example(
        output_path=os.path.join(args.output_dir, 'graph_topology_example.pdf')
    )
    
    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
