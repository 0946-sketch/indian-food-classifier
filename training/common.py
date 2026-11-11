from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA = Path("data/split")
ART = Path("artifacts")
ART.mkdir(exist_ok=True)
IMG = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_model(n_classes, pretrained=True):
    w = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v3_large(weights=w)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    return m


def get_loader(split, tf, shuffle, bs=32):
    ds = datasets.ImageFolder(DATA / split, transform=tf)
    return ds, DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2)
