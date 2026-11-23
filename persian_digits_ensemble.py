import os
import sys
import time
import math
import copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = r"" 
CACHE_FILE = "persian_digits_cache.npz"
BASE_MODEL_PATH = "persian_digits_cnn.pt"
RES_PATH = "member_resnet.pt"
SPEC_PATH = "specialist.pt"
LOG_FILE = "ensemble_log.txt"

SEED = 42                         
VAL_FRAC, TEST_FRAC = 0.10, 0.10

EPOCHS_RES = 8                     
EPOCHS_SPEC = 14                    
GROUP = [2, 3, 4]                   
LABEL_SMOOTHING = 0.05
EMA_DECAY = 0.999
PREVIOUS_RUN_NOTE = "Previous run (base model): test accuracy 99.82%, 27 errors"


torch.set_num_threads(os.cpu_count() or 4)
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


def get_data_dir():
    if DATA_DIR and Path(DATA_DIR).exists():
        return DATA_DIR
    try:
        import kagglehub
        return kagglehub.dataset_download("amirmahdiabbootalebi/persian-handwritten-digits")
    except Exception as e:
        sys.exit(f"Dataset/cache not found. Set DATA_DIR or run the previous script first. ({e})")


def load_one(p):
    im = Image.open(p).convert("L")
    if im.size != (28, 28):
        im = im.resize((28, 28), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def load_dataset():
    if Path(CACHE_FILE).exists():
        d = np.load(CACHE_FILE)
        return d["X"], d["y"]
    root = get_data_dir()
    files = [p for p in Path(root).rglob("*") if p.suffix.lower() in EXTS]
    names = sorted({p.parent.name for p in files})
    cls_idx = {n: i for i, n in enumerate(names)}
    print(f"Reading {len(files)} images ...")
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        X = np.stack(list(ex.map(load_one, files, chunksize=512)))
    y = np.array([cls_idx[p.parent.name] for p in files], dtype=np.int64)
    np.savez_compressed(CACHE_FILE, X=X, y=y)
    return X, y


def stratified_split(y, val_frac, test_frac, seed):
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_te, n_va = int(len(idx) * test_frac), int(len(idx) * val_frac)
        te += idx[:n_te].tolist()
        va += idx[n_te:n_te + n_va].tolist()
        tr += idx[n_te + n_va:].tolist()
    return np.array(tr), np.array(va), np.array(te)


def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Net(nn.Module): 
    def __init__(self, w=32, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            cbr(1, w), cbr(w, w), nn.MaxPool2d(2),
            cbr(w, 2 * w), cbr(2 * w, 2 * w), nn.MaxPool2d(2),
            cbr(2 * w, 4 * w), cbr(4 * w, 4 * w),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * w, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class ResBlock(nn.Module):
    def __init__(self, i, o, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(o)
        self.sc = None if (stride == 1 and i == o) else nn.Sequential(
            nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sc is None else self.sc(x)))


class ResNetS(nn.Module):
    def __init__(self, w=24, n_classes=10):
        super().__init__()
        self.stem = cbr(1, w)
        self.layers = nn.Sequential(ResBlock(w, 2 * w, 2), ResBlock(2 * w, 2 * w),
                                    ResBlock(2 * w, 4 * w, 2), ResBlock(4 * w, 4 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(4 * w, n_classes))

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


def build(kind, width, n_classes):
    return Net(width, n_classes) if kind == "net" else ResNetS(width, n_classes)


class Normed(nn.Module):

    def __init__(self, net, mean, std):
        super().__init__()
        self.net, self.mean, self.std = net, float(mean), float(std)

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def save_member(path, net, kind, width, n_classes, mean, std):
    torch.save({"state_dict": net.state_dict(), "kind": kind, "width": width,
                "n_classes": n_classes, "mean": float(mean), "std": float(std)}, path)


def load_member(path):
    ck = torch.load(path, map_location="cpu")
    net = build(ck.get("kind", "net"), ck["width"], ck.get("n_classes", 10))
    net.load_state_dict(ck["state_dict"])
    return Normed(net, ck["mean"], ck["std"]).eval()


def affine(x, ang_deg, sc, tx, ty):
    ang = ang_deg * math.pi / 180.0
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def augment(x, rot, scl, shf):
    B = x.size(0)

    def r():
        return torch.rand(B) * 2 - 1
    return affine(x, r() * rot, 1 + r() * scl, r() * shf, r() * shf)


TTA = [(0, 1, 0, 0), (0, 1, .07, 0), (0, 1, -.07, 0), (0, 1, 0, .07), (0, 1, 0, -.07), (6, 1, 0, 0), (-6, 1, 0, 0)]


def warp_fixed(x, p):
    B = x.size(0)

    def f(v):
        return torch.full((B,), float(v))
    return affine(x, f(p[0]), f(p[1]), f(p[2]), f(p[3]))


def to01(xb):
    return xb.float().div(255.0).unsqueeze(1)


@torch.no_grad()
def probs_model(model, X, bs=1000):
    model.eval()
    plain, tta = [], []
    for i in range(0, len(X), bs):
        x = to01(X[i:i + bs])
        acc = None
        for j, p in enumerate(TTA):
            pr = F.softmax(model(x if j == 0 else warp_fixed(x, p)), 1)
            if j == 0:
                plain.append(pr)
            acc = pr if acc is None else acc + pr
        tta.append(acc / len(TTA))
    return torch.cat(plain).numpy(), torch.cat(tta).numpy()


class EMA:
    def __init__(self, net, decay):
        self.decay, self.n = decay, 0
        self.shadow = copy.deepcopy(net).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, net):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n)) 
        msd = net.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


@torch.no_grad()
def acc_bare(net, X, y, mean, std, bs=1000):
    net.eval()
    correct = 0
    for i in range(0, len(X), bs):
        correct += (net((to01(X[i:i + bs]) - mean) / std).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(X)


def train_model(name, kind, width, n_classes, Xtr, ytr, Xva, yva, epochs, lr, batch, aug, seed, mean, std, path):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build(kind, width, n_classes)
    ema = EMA(net, EMA_DECAY)
    steps = math.ceil(len(Xtr) / batch)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.2)
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    print(f"\n--- Training {name}: kind={kind} width={width} params={sum(p.numel() for p in net.parameters())} "
          f"train={len(Xtr)} epochs={epochs}")
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(Xtr))
        t0, loss_sum, correct = time.time(), 0.0, 0
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            xb = (augment(to01(Xtr[idx]), *aug) - mean) / std
            yb = ytr[idx]
            opt.zero_grad(set_to_none=True)
            out = net(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            ema.update(net)
            loss_sum += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
        va = acc_bare(ema.shadow, Xva, yva, mean, std)
        if va >= best:
            best, best_state = va, {k: v.clone() for k, v in ema.shadow.state_dict().items()}
        print(f"[{name}] epoch {ep:2d}/{epochs} | loss {loss_sum / len(Xtr):.4f} | train acc {correct / len(Xtr):.4f} "
              f"| val acc (EMA) {va:.4f} | {time.time() - t0:.0f}s", flush=True)
    ema.shadow.load_state_dict(best_state)
    save_member(path, ema.shadow, kind, width, n_classes, mean, std)
    print(f"[{name}] best val acc {best:.4f} -> saved {path}")


def nll(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1.0)).mean())


def acc(P, y):
    return float((P.argmax(1) == y).mean())


def combine(P, S, idx, G, alpha):
    P2 = P.copy()
    if len(idx) == 0 or alpha == 0:
        return P2
    G = np.array(G)
    mass = P[idx][:, G].sum(1, keepdims=True)
    base = P[idx][:, G] / np.clip(mass, 1e-9, None)
    P2[idx[:, None], G[None, :]] = ((1 - alpha) * base + alpha * S) * mass
    return P2


def save_error_grid(X_np, y_true, y_pred, path, max_n=60):
    idx = np.where(y_true != y_pred)[0][:max_n]
    if len(idx) == 0:
        return
    cols, cw, ch = 10, 84, 104
    rows = math.ceil(len(idx) / cols)
    canvas = Image.new("L", (cols * cw, rows * ch), 255)
    d = ImageDraw.Draw(canvas)
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        canvas.paste(Image.fromarray(X_np[i]).resize((84, 84), Image.BICUBIC), (c * cw, r * ch))
        d.text((c * cw + 4, r * ch + 88), f"T:{y_true[i]} P:{y_pred[i]}", fill=0)
    canvas.save(path)


def confusion(y, p, n=10):
    cm = np.zeros((n, n), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    return cm


def main():
    sys.stdout = Tee(LOG_FILE)
    t_start = time.time()

    X, y = load_dataset()
    if X.mean() > 127:
        X = 255 - X
    tr, va, te = stratified_split(y, VAL_FRAC, TEST_FRAC, SEED)
    X_te_np, y_te_np, y_va_np = X[te], y[te], y[va]
    Xtr, ytr = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
    Xva, yva = torch.from_numpy(X[va]), torch.from_numpy(y[va])
    Xte = torch.from_numpy(X_te_np)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")
    f = to01(Xtr[:20000])
    mean, std = f.mean().item(), f.std().item()
    G = np.array(GROUP)

    if not Path(BASE_MODEL_PATH).exists():
        sys.exit(f"{BASE_MODEL_PATH} not found. Run persian_digits_cnn.py first (in this same folder).")
    base = load_member(BASE_MODEL_PATH)

    if not Path(RES_PATH).exists():
        train_model("resnet", "res", 24, 10, Xtr, ytr, Xva, yva, EPOCHS_RES, lr=3e-3, batch=256,
                    aug=(12, 0.12, 0.14), seed=SEED + 1, mean=mean, std=std, path=RES_PATH)
    res = load_member(RES_PATH)

    if not Path(SPEC_PATH).exists():
        lut = -np.ones(10, dtype=np.int64)
        lut[G] = np.arange(len(G))
        m_tr, m_va = np.isin(y[tr], G), np.isin(y[va], G)
        Xs_tr, ys_tr = Xtr[torch.from_numpy(m_tr)], torch.from_numpy(lut[y[tr][m_tr]])
        Xs_va, ys_va = Xva[torch.from_numpy(m_va)], torch.from_numpy(lut[y[va][m_va]])
        train_model("specialist", "net", 32, len(G), Xs_tr, ys_tr, Xs_va, ys_va, EPOCHS_SPEC, lr=3e-3, batch=128,
                    aug=(14, 0.14, 0.16), seed=SEED + 2, mean=mean, std=std, path=SPEC_PATH)
    spec = load_member(SPEC_PATH)

    print("\nComputing predictions (with TTA) ...", flush=True)
    pr = {}
    for name, m in [("base", base), ("resnet", res)]:
        t0 = time.time()
        pr[name] = {"val": probs_model(m, Xva), "test": probs_model(m, Xte)}
        print(f"  {name}: done in {time.time() - t0:.0f}s", flush=True)

    best = None
    for use_tta in (0, 1):
        for w in np.arange(0, 1.01, 0.1):
            P = w * pr["base"]["val"][use_tta] + (1 - w) * pr["resnet"]["val"][use_tta]
            s = nll(P, y_va_np)
            if best is None or s < best[0]:
                best = (s, use_tta, float(w))
    _, use_tta, w_base = best
    print(f"\nChosen on validation: TTA={'on' if use_tta else 'off'}, weight(base)={w_base:.1f}, weight(resnet)={1 - w_base:.1f}")

    def ens(split):
        return w_base * pr["base"][split][use_tta] + (1 - w_base) * pr["resnet"][split][use_tta]

    P_va, P_te = ens("val"), ens("test")

    gate_va = np.where(P_va[:, G].sum(1) > 0.5)[0]
    gate_te = np.where(P_te[:, G].sum(1) > 0.5)[0]
    S_va = probs_model(spec, Xva[torch.from_numpy(gate_va)])[use_tta]
    S_te = probs_model(spec, Xte[torch.from_numpy(gate_te)])[use_tta]
    print(f"Specialist applied to {len(gate_va)} val and {len(gate_te)} test samples (predicted in {GROUP})")

    best_a = None
    print("alpha (specialist weight) -> val NLL / val acc")
    for a in np.arange(0, 1.01, 0.1):
        P2 = combine(P_va, S_va, gate_va, G, a)
        s = nll(P2, y_va_np)
        print(f"  {a:.1f}: {s:.5f} / {acc(P2, y_va_np) * 100:.2f}%")
        if best_a is None or s < best_a[0] - 1e-9:
            best_a = (s, float(a))
    alpha = best_a[1]
    print(f"Chosen alpha = {alpha:.1f}")
    P_te_final = combine(P_te, S_te, gate_te, G, alpha)
    P_va_final = combine(P_va, S_va, gate_va, G, alpha)

    def n_err(P, yy):
        return int((P.argmax(1) != yy).sum())

    def n_err_group(P, yy):
        m = np.isin(yy, G)
        return int((P.argmax(1)[m] != yy[m]).sum())

    rows = [
        ("Base CNN (plain)", pr["base"]["val"][0], pr["base"]["test"][0]),
        ("Base CNN + TTA", pr["base"]["val"][1], pr["base"]["test"][1]),
        ("ResNet (plain)", pr["resnet"]["val"][0], pr["resnet"]["test"][0]),
        ("ResNet + TTA", pr["resnet"]["val"][1], pr["resnet"]["test"][1]),
        ("Ensemble (chosen settings)", P_va, P_te),
        ("Ensemble + Specialist (FINAL)", P_va_final, P_te_final),
    ]
    print("\n" + "=" * 86)
    print(f"{'Model':32s} {'val acc':>8s} {'test acc':>9s} {'test err':>9s} {'err in ' + str(GROUP):>14s}")
    for nm, pv, pt in rows:
        print(f"{nm:32s} {acc(pv, y_va_np) * 100:7.2f}% {acc(pt, y_te_np) * 100:8.2f}% {n_err(pt, y_te_np):9d} "
              f"{n_err_group(pt, y_te_np):14d}")
    print("=" * 86)
    print(PREVIOUS_RUN_NOTE + "  (should match the Base CNN (plain) row; if not, the data split differs)")

    pred_final = P_te_final.argmax(1)
    cm = confusion(y_te_np, pred_final)
    print("\nPer-class accuracy (FINAL):")
    for c in range(10):
        print(f"  class {c}: {cm[c, c] / max(cm[c].sum(), 1) * 100:.2f}%   errors={int(cm[c].sum() - cm[c, c])}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)
    off = [(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
    off.sort(reverse=True)
    print("\nMost frequent confusions (count: true -> pred):")
    for cnt, i, j in off[:8]:
        print(f"  {cnt}: {i} -> {j}")

    np.savez_compressed("ensemble_probs.npz", P_test=P_te_final, y_test=y_te_np, P_val=P_va_final, y_val=y_va_np)
    save_error_grid(X_te_np, y_te_np, pred_final, "errors_grid.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm0 = confusion(y_te_np, pr["base"]["test"][0].argmax(1))
        e0 = [int(cm0[c].sum() - cm0[c, c]) for c in range(10)]
        e1 = [int(cm[c].sum() - cm[c, c]) for c in range(10)]
        xs = np.arange(10)
        plt.figure(figsize=(8, 4))
        plt.bar(xs - 0.2, e0, 0.4, label="Base CNN")
        plt.bar(xs + 0.2, e1, 0.4, label="Hybrid (final)")
        plt.xticks(xs)
        plt.xlabel("class")
        plt.ylabel("test errors")
        plt.title("Errors per class")
        plt.legend()
        plt.tight_layout()
        plt.savefig("errors_per_class.png", dpi=150)
        plt.close()
    except Exception as e:
        print("(matplotlib not available, skipped plot):", e)

    print(f"\nSaved: {LOG_FILE}, ensemble_probs.npz, errors_grid.png, errors_per_class.png, {RES_PATH}, {SPEC_PATH}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
