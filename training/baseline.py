"""Baseline: frozen pretrained features + logistic regression."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torchvision import models
from common import ART, DEVICE, eval_tf, get_loader

m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
m.classifier = nn.Identity()  # outputs 960-d pooled features
m.eval().to(DEVICE)


def feats(split):
    _, dl = get_loader(split, eval_tf, False)
    X, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            X.append(m(x.to(DEVICE)).cpu().numpy())
            Y.append(y.numpy())
    return np.concatenate(X), np.concatenate(Y)


Xtr, ytr = feats("train")
Xte, yte = feats("test")
clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
acc = accuracy_score(yte, clf.predict(Xte))
print(f"Baseline (frozen features + logistic regression) test accuracy: {acc:.4f}")
(ART / "baseline.txt").write_text(f"baseline_test_accuracy={acc:.4f}\n")
