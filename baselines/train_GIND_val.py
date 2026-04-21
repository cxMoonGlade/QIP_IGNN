"""
GIND with the QIGNN paper protocol for TU graph classification:
- Stratified 10-fold CV with an internal validation split
- Best validation accuracy for model selection
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
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree

THIS_DIR = osp.dirname(osp.realpath(__file__))
PROJECT_DIR = osp.dirname(THIS_DIR)
WORKSPACE_DIR = osp.dirname(PROJECT_DIR)
GIND_ROOT = os.environ.get("GIND_ROOT", osp.join(WORKSPACE_DIR, "out_resources", "GIND"))

if GIND_ROOT not in sys.path:
    sys.path.insert(0, GIND_ROOT)

from libs.normalization import cal_norm
from libs.utils import set_seed
from model.gind import GIND


def separate_data_with_val(labels, seed, fold_idx):
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


def ensure_features(data):
    if data.x is None:
        deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float32)
        data.x = deg.unsqueeze(-1)
    return data


def evaluate(model, loader, device, add_self_loops):
    model.eval()
    correct = 0
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for data in loader:
            data = ensure_features(data).to(device)
            norm_factor, edge_index = cal_norm(
                data.edge_index, data.num_nodes, self_loop=add_self_loops
            )
            output = model(data.x, edge_index, norm_factor, batch=data.batch)
            loss = F.cross_entropy(output, data.y)
            total_loss += loss.item() * data.num_graphs
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            total += data.num_graphs

    return correct / total * 100.0, total_loss / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="NCI1")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dropout_imp", type=float, default=0.0)
    parser.add_argument("--dropout_exp", type=float, default=0.4)
    parser.add_argument("--drop_input", action="store_true")
    parser.add_argument("--iter_total", type=int, default=16)
    parser.add_argument("--iter_grad", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--result_dir", type=str, default="results_val")
    parser.add_argument("--add_self_loops", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, device)

    path = osp.join(PROJECT_DIR, "data_implicit_baselines", args.dataset)
    dataset = TUDataset(path, name=args.dataset)
    labels = [data.y.item() for data in dataset]

    train_idx, val_idx, test_idx = separate_data_with_val(labels, args.seed, args.fold_idx)
    train_dataset = [ensure_features(dataset[i]) for i in train_idx]
    val_dataset = [ensure_features(dataset[i]) for i in val_idx]
    test_dataset = [ensure_features(dataset[i]) for i in test_idx]

    sample = train_dataset[0]
    in_channels = sample.x.size(-1)
    out_channels = dataset.num_classes

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    print(f"GIND on {args.dataset} | fold {args.fold_idx} seed {args.seed}")
    print(f"Train {len(train_idx)}, Val {len(val_idx)}, Test {len(test_idx)}")

    model = GIND(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        alpha=args.alpha,
        iter_nums=(args.iter_total, args.iter_grad),
        dropout_imp=args.dropout_imp,
        dropout_exp=args.dropout_exp,
        drop_input=args.drop_input,
        norm="InstanceNorm",
        residual=True,
        rescale=True,
        linear=False,
        double_linear=True,
        act_imp="tanh",
        act_exp="elu",
        final_reduce="add",
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params}")

    best_val_acc = -1.0
    best_val_epoch = 0
    best_state = None
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_all = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            norm_factor, edge_index = cal_norm(
                data.edge_index, data.num_nodes, self_loop=args.add_self_loops
            )
            output = model(data.x, edge_index, norm_factor, batch=data.batch)
            loss = F.cross_entropy(output, data.y)
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * data.num_graphs

        train_loss = loss_all / len(train_idx)
        train_acc, _ = evaluate(model, train_loader, device, args.add_self_loops)
        val_acc, val_loss = evaluate(model, val_loader, device, args.add_self_loops)
        epoch_time = time.time() - t0

        print(
            f"epoch {epoch:3d} | loss {train_loss:.4f} | "
            f"train {train_acc:.2f}% | val {val_acc:.2f}% val_loss {val_loss:.4f} | "
            f"time {epoch_time:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "time": epoch_time,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, test_loss = evaluate(model, test_loader, device, args.add_self_loops)
    total_time = time.time() - start_time

    print(f"\nBest Val: {best_val_acc:.2f}% (epoch {best_val_epoch})")
    print(f"Test Acc: {test_acc:.2f}%")
    print(f"Time: {total_time:.1f}s")

    result_dir = osp.join(args.result_dir, args.dataset)
    os.makedirs(result_dir, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M")
    result = {
        "config": vars(args),
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "n_params": n_params,
        "training_time": {"total_seconds": total_time},
        "history": history,
    }
    path = osp.join(result_dir, f"result_fold{args.fold_idx}_seed{args.seed}_{ts}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
