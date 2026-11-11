"""Train / evaluate VTMT-Net.   python train.py --epochs 3 --n-train 32 --n-val 16

Synthetic smoke-test by default. For RSNA-STR, build RSNAStudyDataset (dataset.py) and
pass n_loc=3 (left, right, central).
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SyntheticPEDataset
from losses import MultiTaskLoss
from model import VTMTNet, burden_index
from synthetic import N_LOC


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = s.argsort().argsort() + 1.0
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def to_dev(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    y, s, dices, bi_p, bi_t, sev_y, sev_s = [], [], [], [], [], [], []
    for b in loader:
        b = to_dev(b, dev)
        o = model(b["image"], b["nodes"], b["adj"], b["node_mask"])
        y += b["det"].tolist(); s += torch.sigmoid(o["det"]).tolist()
        sev_y += b["rvlv_hi"].tolist(); sev_s += torch.sigmoid(o["rvlv"]).tolist()
        pred = (o["p"] > 0.5).float(); t = b["seg"]
        inter = (pred * t).flatten(1).sum(1); den = pred.flatten(1).sum(1) + t.flatten(1).sum(1)
        pos = t.flatten(1).sum(1) > 0
        dices += ((2 * inter + 1e-6) / (den + 1e-6))[pos].tolist()
        bi_p += burden_index(torch.sigmoid(o["node_logit"]), b["nodes"], b["node_mask"]).tolist()
        bi_t += b["oi"].tolist()
    ok = [i for i, v in enumerate(bi_t) if v >= 0]
    return {"auc_det": auroc(y, s), "auc_severity": auroc(sev_y, sev_s),
            "dice_pos": float(np.mean(dices)) if dices else float("nan"),
            "bi_corr": float(np.corrcoef(np.array(bi_p)[ok], np.array(bi_t)[ok])[0, 1]) if len(ok) > 2 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weak-fraction", type=float, default=0.5, help="share of train studies with hidden masks")
    ap.add_argument("--no-topology", action="store_true")
    ap.add_argument("--no-multiscale", action="store_true")
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="vtmt.pt")
    a = ap.parse_args()

    dev = torch.device(a.device)
    tr = DataLoader(SyntheticPEDataset(a.n_train, a.size, 128, seed=1, weak_fraction=a.weak_fraction),
                    batch_size=a.bs, shuffle=True, num_workers=0)
    va = DataLoader(SyntheticPEDataset(a.n_val, a.size, 128, seed=2), batch_size=a.bs)
    model = VTMTNet(in_ch=a.in_ch, n_loc=N_LOC, use_topology=not a.no_topology,
                    use_multiscale=not a.no_multiscale).to(dev)
    crit = MultiTaskLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for b in tr:
            b = to_dev(b, dev)
            img = b["image"][:, : a.in_ch]
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                o = model(img, b["nodes"], b["adj"], b["node_mask"])
            loss, parts = crit(o, dict(b, image=b["image"]))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tot += float(loss)
        print(f"epoch {ep + 1}: loss {tot / len(tr):.4f}  last parts {parts}")
        print("  val", evaluate(model, va, dev))
    torch.save(model.state_dict(), a.out)


if __name__ == "__main__":
    main()
