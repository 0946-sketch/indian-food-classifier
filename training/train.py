"""Two-phase transfer learning with MobileNetV3-Large, then ONNX export."""
import json
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ART, DEVICE, build_model, eval_tf, get_loader, train_tf

E1, E2 = 5, 8          # epochs: phase 1 (head only), phase 2 (fine-tune)
LR1, LR2 = 1e-3, 1e-4  # phase 2 uses a lower learning rate

train_ds, train_dl = get_loader("train", train_tf, True)
val_ds, val_dl = get_loader("val", eval_tf, False)
classes = train_ds.classes
(ART / "classes.json").write_text(json.dumps(classes))
print(f"{len(classes)} classes, device={DEVICE}")

model = build_model(len(classes)).to(DEVICE)
crit = nn.CrossEntropyLoss()
hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best = {"acc": 0.0}


def run_epoch(dl, opt=None):
    training = opt is not None
    model.train(training)
    tot_loss = tot_ok = n = 0
    with torch.set_grad_enabled(training):
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = crit(out, y)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot_loss += loss.item() * len(y)
            tot_ok += (out.argmax(1) == y).sum().item()
            n += len(y)
    return tot_loss / n, tot_ok / n


def run_phase(name, epochs, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        tl, ta = run_epoch(train_dl, opt)
        vl, va = run_epoch(val_dl)
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        print(f"[{name}] epoch {ep+1}/{epochs} train_loss={tl:.3f} train_acc={ta:.3f} val_loss={vl:.3f} val_acc={va:.3f}")
        if va > best["acc"]:
            best["acc"] = va
            torch.save(model.state_dict(), ART / "best.pt")


# Phase 1: freeze backbone, train only the new classification head
model.features.requires_grad_(False)
run_phase("phase1-head", E1, LR1)

# Phase 2: unfreeze the top layers and fine-tune with a lower LR
for p in model.features[-4:].parameters():
    p.requires_grad = True
run_phase("phase2-finetune", E2, LR2)

# Log curves
(ART / "history.json").write_text(json.dumps(hist))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, key, title in zip(ax, ("loss", "acc"), ("Loss", "Accuracy")):
    a.plot(hist[f"train_{key}"], label="train")
    a.plot(hist[f"val_{key}"], label="val")
    a.axvline(E1 - 0.5, color="gray", linestyle="--", label="unfreeze")
    a.set_title(title); a.set_xlabel("epoch"); a.legend()
plt.tight_layout()
plt.savefig(ART / "curves.png", dpi=150)

# Export best weights to ONNX (small + fast on free-tier CPUs)
model.load_state_dict(torch.load(ART / "best.pt", map_location="cpu"))
model.eval().cpu()
dummy = torch.randn(1, 3, 224, 224)
kwargs = dict(input_names=["input"], output_names=["logits"],
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
try:
    torch.onnx.export(model, dummy, str(ART / "model.onnx"), dynamo=False, **kwargs)
except TypeError:
    torch.onnx.export(model, dummy, str(ART / "model.onnx"), **kwargs)
print(f"Best val acc: {best['acc']:.3f}. Saved artifacts/model.onnx, classes.json, curves.png")
