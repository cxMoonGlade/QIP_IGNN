"""
IGNN (SwiftieH/IGNN graphclassification) on OGB molecular graphs.

Expects a full IGNN clone at IGNN_DIR (see run_ignn_gind_baselines.sh), e.g.:
  <workspace>/out_resources/IGNN-main/graphclassification

Protocol matches train_GIND_ogb: scaffold split by default, or stratified 10-fold
(--n_folds 10 --fold_idx k) for parity with TU runs.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader

THIS_DIR = osp.dirname(osp.realpath(__file__))
PROJECT_DIR = osp.dirname(THIS_DIR)
WORKSPACE_DIR = osp.dirname(PROJECT_DIR)
IGNN_DIR = os.environ.get(
    "IGNN_DIR", osp.join(WORKSPACE_DIR, "out_resources", "IGNN-main", "graphclassification")
)
if IGNN_DIR not in sys.path:
    sys.path.insert(0, IGNN_DIR)

from models import IGNN  # noqa: E402
from normalization import fetch_normalization  # noqa: E402


def separate_data_with_val(labels: list[int], seed: int, fold_idx: int):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    idx_list = list(skf.split(np.zeros(len(labels)), labels))
    train_idx, test_idx = idx_list[fold_idx]

    train_labels = [labels[i] for i in train_idx]
    val_skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=seed)
    sub_splits = list(val_skf.split(np.zeros(len(train_idx)), train_labels))
    sub_train_idx, sub_val_idx = sub_splits[0]

    final_train = train_idx[sub_train_idx]
    final_val = train_idx[sub_val_idx]
    return final_train.tolist(), final_val.tolist(), test_idx.tolist()


def build_adj_tensor(data, device: torch.device):
    """Symmetric AugNormAdj sparse tensor (matches IGNN train_IGNN.py test path)."""
    edge_weight = torch.ones(
        (data.edge_index.size(1),), dtype=torch.float32, device=data.edge_index.device
    )
    adj_sp = csr_matrix(
        (
            edge_weight.cpu().numpy(),
            (data.edge_index[0].cpu().numpy(), data.edge_index[1].cpu().numpy()),
        ),
        shape=(data.num_nodes, data.num_nodes),
    )
    adj_sp = adj_sp + adj_sp.T
    adj_normalizer = fetch_normalization("AugNormAdj")
    adj_sp_nz = adj_normalizer(adj_sp)
    idx = torch.LongTensor(np.array([adj_sp_nz.row, adj_sp_nz.col])).to(device)
    val = torch.Tensor(adj_sp_nz.data).to(device)
    adj = torch.sparse.FloatTensor(idx, val, torch.Size([data.num_nodes, data.num_nodes]))
    return adj


class IGNNOGB(nn.Module):
    def __init__(self, hidden: int, kappa: float, dropout: float):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden)
        self.ignn = IGNN(
            nfeat=hidden,
            nhid=hidden,
            nclass=2,
            num_node=None,
            dropout=dropout,
            kappa=kappa,
        )

    def forward(self, data, adj):
        x = self.atom_encoder(data.x)
        return self.ignn(x.T, adj, data.batch)


def train_epoch(model, loader, optimizer, device, add_self_loops: bool):
    del add_self_loops  # IGNN uses fixed normalizer; kept for CLI parity with GIND wrapper
    model.train()
    total_loss = 0.0
    total = 0

    for data in loader:
        data = data.to(device)
        adj = build_adj_tensor(data, device)
        optimizer.zero_grad()
        out = model(data, adj)
        y = data.y.view(-1).long().clamp(min=0, max=1)
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total += data.num_graphs

    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate_auc(model, loader, evaluator, device):
    model.eval()
    y_true, y_pred = [], []

    for data in loader:
        data = data.to(device)
        adj = build_adj_tensor(data, device)
        out = model(data, adj)
        prob = out.exp()[:, 1].view(-1, 1)
        y_true.append(data.y.view(-1, 1).float().cpu())
        y_pred.append(prob.cpu())

    y_true_t = torch.cat(y_true, dim=0).numpy()
    y_pred_t = torch.cat(y_pred, dim=0).numpy()
    return evaluator.eval({"y_true": y_true_t, "y_pred": y_pred_t})["rocauc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--kappa", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=0)
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--result_dir", type=str, default="results_ogb")
    parser.add_argument("--add_self_loops", action="store_true")
    args = parser.parse_args()

    if not osp.isfile(osp.join(IGNN_DIR, "models.py")):
        print(f"IGNN graphclassification not found at IGNN_DIR={IGNN_DIR!r}", file=sys.stderr)
        sys.exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading {args.dataset} (IGNN_DIR={IGNN_DIR})...")
    dataset = PygGraphPropPredDataset(
        name=args.dataset, root=osp.join(PROJECT_DIR, "data_implicit_baselines")
    )
    evaluator = Evaluator(name=args.dataset)

    if args.n_folds > 0:
        if args.n_folds != 10:
            raise ValueError("Only n_folds=10 is supported (TU protocol).")
        assert 0 <= args.fold_idx < args.n_folds
        ds_list = [dataset[i] for i in range(len(dataset))]
        labels = [int(d.y.view(-1)[0].item()) for d in ds_list]
        tr_i, va_i, te_i = separate_data_with_val(labels, args.seed, args.fold_idx)
        train_loader = DataLoader([ds_list[i] for i in tr_i], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader([ds_list[i] for i in va_i], batch_size=args.batch_size)
        test_loader = DataLoader([ds_list[i] for i in te_i], batch_size=args.batch_size)
        print(
            f"  10-fold CV fold {args.fold_idx}: train {len(tr_i)}, val {len(va_i)}, test {len(te_i)}"
        )
    else:
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(
            dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size)

    model = IGNNOGB(hidden=args.hidden, kappa=args.kappa, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"IGNN Params: {n_params}")

    best_val_auc = -1.0
    best_epoch = 0
    best_state = None
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, args.add_self_loops)
        val_auc = evaluate_auc(model, val_loader, evaluator, device)
        epoch_time = time.time() - t0

        print(
            f"epoch {epoch:3d} | loss {train_loss:.4f} | val AUC {val_auc:.4f} | time {epoch_time:.1f}s"
        )

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc, "time": epoch_time}
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_auc = evaluate_auc(model, test_loader, evaluator, device)
    total_time = time.time() - start_time

    print(f"\nBest Val AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"Test AUC:     {test_auc:.4f}")
    print(f"Time:         {total_time:.1f}s")

    result_dir = osp.join(args.result_dir, args.dataset)
    os.makedirs(result_dir, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M")
    result = {
        "config": vars(args),
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "test_auc": test_auc,
        "n_params": n_params,
        "training_time": {"total_seconds": total_time},
        "history": history,
    }
    if args.n_folds > 0:
        path = osp.join(result_dir, f"result_fold{args.fold_idx}_seed{args.seed}_{ts}.json")
    else:
        path = osp.join(result_dir, f"result_seed{args.seed}_{ts}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
