"""Test-set only: accuracy, per-class precision/recall, confusion matrix, wrong predictions."""
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from common import ART, DEVICE, build_model, eval_tf, get_loader

ds, dl = get_loader("test", eval_tf, False)  # shuffle=False keeps order aligned with ds.samples
classes = ds.classes
model = build_model(len(classes), pretrained=False)
model.load_state_dict(torch.load(ART / "best.pt", map_location=DEVICE))
model.eval().to(DEVICE)

probs = []
with torch.no_grad():
    for x, _ in dl:
        probs.append(torch.softmax(model(x.to(DEVICE)), 1).cpu().numpy())
probs = np.concatenate(probs)
y_true = np.array([s[1] for s in ds.samples])
y_pred = probs.argmax(1)

acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=classes, digits=3)
print(f"Test accuracy: {acc:.4f}\n{report}")
(ART / "test_report.txt").write_text(f"Test accuracy: {acc:.4f}\n\n{report}")

cm = confusion_matrix(y_true, y_pred)
size = max(6, len(classes) * 0.6)
fig, ax = plt.subplots(figsize=(size, size))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90)
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
for i in range(len(classes)):
    for j in range(len(classes)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
plt.tight_layout()
plt.savefig(ART / "confusion_matrix.png", dpi=150)

# Wrong predictions -> CSV (add your own 'why' column!) + image grid
wrong = np.where(y_true != y_pred)[0]
with open(ART / "wrong_predictions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path", "true", "predicted", "confidence", "why_wrong (fill in)"])
    for i in wrong:
        w.writerow([ds.samples[i][0], classes[y_true[i]], classes[y_pred[i]], f"{probs[i].max():.3f}", ""])
show = wrong[:12]
if len(show):
    cols = 4
    rows = (len(show) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()
    for a in axes:
        a.axis("off")
    for a, i in zip(axes, show):
        a.imshow(Image.open(ds.samples[i][0]))
        a.set_title(f"true: {classes[y_true[i]]}\npred: {classes[y_pred[i]]}", fontsize=8)
    plt.tight_layout()
    plt.savefig(ART / "wrong_predictions.png", dpi=150)
print(f"{len(wrong)} wrong predictions saved. Analyse at least 10 of them.")
