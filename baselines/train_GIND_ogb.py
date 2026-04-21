"""
GIND with the QIGNN paper protocol for OGB graph property prediction:
- Default: official scaffold split (--n_folds 0)
- Optional: stratified 10-fold CV + internal val (--n_folds 10 --fold_idx k), same as TU
- Best validation ROC-AUC for model selection
- JSON result output
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
import torch.nn.functional as F
import torch.optim as optim
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader

THIS_DIR = osp.dirname(osp.realpath(__file__))
PROJECT_DIR = osp.dirname(THIS_DIR)
WORKSPACE_DIR = osp.dirname(PROJECT_DIR)
GIND_ROOT = os.environ.get("GIND_ROOT", osp.join(WORKSPACE_DIR, "out_resources", "GIND"))

if GIND_ROOT not in sys.path:
    sys.path.insert(0, GIND_ROOT)

from libs.normalization import cal_norm
from libs.utils import set_seed
from model.gind import GIND


_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{**kwargs, "weights_only": False})


def separate_data_with_val(labels: list[int], seed: int, fold_idx: int):
    """Match train_GIND_val / QIGNN train.py k-fold protocol."""
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


class GINDForBinary(torch.nn.Module):
    def __init__(
        self,
        hidden: int,
        num_layers: int,
        alpha: float,
        iter_total: int,
        iter_grad: int,
        dropout_imp: float,
        dropout_exp: float,
        drop_input: bool,
    ):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden)
        self.gind = GIND(
            in_channels=hidden,
            out_channels=1,
            hidden_channels=hidden,
            num_layers=num_layers,
            alpha=alpha,
            iter_nums=(iter_total, iter_grad),
            dropout_imp=dropout_imp,
            dropout_exp=dropout_exp,
            drop_input=drop_input,
            norm="InstanceNorm",
            residual=True,
            rescale=True,
            linear=True,
            double_linear=True,
            act_imp="tanh",
            act_exp="elu",
            final_reduce="add",
        )

    def forward(self, data, edge_index, norm_factor):
        x = self.atom_encoder(data.x)
        return self.gind(x, edge_index, norm_factor, batch=data.batch)


def train_epoch(model, loader, optimizer, device, add_self_loops):
    model.train()
    total_loss = 0.0
    total = 0

    for data in loader:
        data = data.to(device)
        norm_factor, edge_index = cal_norm(
            data.edge_index, data.num_nodes, self_loop=add_self_loops
        )
        optimizer.zero_grad()
        logits = model(data, edge_index, norm_factor).view(-1)
        labels = data.y.view(-1).float()
        is_labeled = labels == labels
        loss = F.binary_cross_entropy_with_logits(logits[is_labeled], labels[is_labeled])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total += data.num_graphs

    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, evaluator, device, add_self_loops):
    model.eval()
    y_true, y_pred = [], []

    for data in loader:
        data = data.to(device)
        norm_factor, edge_index = cal_norm(
            data.edge_index, data.num_nodes, self_loop=add_self_loops
        )
        logits = model(data, edge_index, norm_factor)
        y_true.append(data.y.view(-1, 1).cpu())
        y_pred.append(logits.view(-1, 1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dropout_imp", type=float, default=0.0)
    parser.add_argument("--dropout_exp", type=float, default=0.4)
    parser.add_argument("--drop_input", action="store_true")
    parser.add_argument("--iter_total", type=int, default=16)
    parser.add_argument("--iter_grad", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=0,
                        help="0 = OGB scaffold split; >0 = stratified k-fold on full dataset")
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--result_dir", type=str, default="results_ogb")
    parser.add_argument("--add_self_loops", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, device)

    print(f"Loading {args.dataset}...")
    dataset = PygGraphPropPredDataset(
        name=args.dataset, root=osp.join(PROJECT_DIR, "data_implicit_baselines")
    )
    evaluator = Evaluator(name=args.dataset)

    if args.n_folds > 0:
        if args.n_folds != 10:
            raise ValueError("OGB k-fold uses the same protocol as TU; only n_folds=10 is supported.")
        assert 0 <= args.fold_idx < args.n_folds
        ds_list = [dataset[i] for i in range(len(dataset))]
        labels = [int(d.y.view(-1)[0].item()) for d in ds_list]
        tr_i, va_i, te_i = separate_data_with_val(labels, args.seed, args.fold_idx)
        train_loader = DataLoader([ds_list[i] for i in tr_i], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader([ds_list[i] for i in va_i], batch_size=args.batch_size)
        test_loader = DataLoader([ds_list[i] for i in te_i], batch_size=args.batch_size)
        print(
            f"  {args.n_folds}-fold CV fold {args.fold_idx}: "
            f"train {len(tr_i)}, val {len(va_i)}, test {len(te_i)}"
        )
    else:
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size)

    model = GINDForBinary(
        hidden=args.hidden,
        num_layers=args.num_layers,
        alpha=args.alpha,
        iter_total=args.iter_total,
        iter_grad=args.iter_grad,
        dropout_imp=args.dropout_imp,
        dropout_exp=args.dropout_exp,
        drop_input=args.drop_input,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GIND Params: {n_params}")

    best_val_auc = -1.0
    best_epoch = 0
    best_state = None
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, args.add_self_loops)
        val_auc = evaluate(model, val_loader, evaluator, device, args.add_self_loops)
        epoch_time = time.time() - t0

        print(
            f"epoch {epoch:3d} | loss {train_loss:.4f} | "
            f"val AUC {val_auc:.4f} | time {epoch_time:.1f}s"
        )

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc, "time": epoch_time}
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_auc = evaluate(model, test_loader, evaluator, device, args.add_self_loops)
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
