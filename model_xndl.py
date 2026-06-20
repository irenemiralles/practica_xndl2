# Laboratori XNDL final notat - classificacio d'imatges 32x32 en 14 categories
#
# Model entrenat DES DE ZERO (cap pes preentrenat). CNN en lloc d'una xarxa
# fully-connected: explota l'estructura espacial i te molts menys parametres
# (connectivitat esparsa + comparticio de pesos).
#
# Metodologia (sense data leakage):
#   - El 'val' nomes s'usa per a l'avaluacio FINAL.
#   - Seleccio de model amb hold-out estratificat tret del 'train' (dev).
#     Nomes desem el model quan el dev millora (descartem epoques pitjors).

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# ----- Hiperparametres -----
DATA_DIR        = "./dades"   # ../dades/{train,val,test}
BATCH_SIZE      = 256          # 32x32 ocupen poc -> batch gran = passos estables i rapids
EPOCHS          = 20
LR              = 1e-3         # Adam, LR adaptatiu (apunts 14.3)
WEIGHT_DECAY    = 1e-4         # regularitzacio L2 suau; el gap train/dev no mostra overfitting
DROPOUT         = 0.2          # regularitzacio moderada (apunts 14.4)
DEV_FRACTION    = 0.10         # fraccio del train reservada per seleccionar model
NUM_WORKERS     = 4            # paral.lelitza l'augmentacio (NO es teoria, es velocitat). Windows: si peta, 0
SEED            = 42


# Precarreguem cada split a memoria (uint8) un sol cop: l'I/O de disc (112k
# PNGs) era el coll d'ampolla; despres cada epoca es independent del disc.
def preload(folder, workers):
    ds = datasets.ImageFolder(folder, transform=transforms.Compose([
        transforms.Grayscale(), transforms.PILToTensor()]))  # uint8 [1,32,32]
    loader = DataLoader(ds, batch_size=1024, num_workers=workers)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)
    return torch.cat(xs), torch.cat(ys), ds.classes


class MemDS(Dataset):
    def __init__(self, X, Y, mean, std, train):
        self.X, self.Y = X, Y
        aug = []
        if train:
            # Data augmentation (apunts 14.4). Crop amb padding + flip: barat
            # (nomes pad+tall, sense grid_sample) i dona invariancia a petites
            # translacions sense penalitzar el temps per epoca.
            aug = [transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(0.5)]
        self.tf = transforms.Compose(
            [transforms.ConvertImageDtype(torch.float32)] + aug +   # uint8 -> [0,1]
            [transforms.Normalize((mean,), (std,))])                # normalitza amb stats reals

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.tf(self.X[i]), int(self.Y[i])


# CNN tipus VGG, piramide (apunts sec. 9): mida espacial baixa (pooling),
# canals pugen. Filtres 3x3 (apilats > un filtre gran, apunts 7.1). Canals
# 64/128/256: ampliem capacitat perque el dev >= train indicava underfitting.
class SmallCNN(nn.Module):
    def __init__(self, n_classes, in_ch=1, p_drop=0.2):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in,  c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),   # BatchNorm (14.4) + ReLU no saturada (14.2)
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                                 # invariancia a petites translacions (8.3)
            )

        self.features = nn.Sequential(
            block(in_ch, 64),    # 32 -> 16
            block(64, 128),      # 16 -> 8
            block(128, 256),     # 8  -> 4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # global average pooling (8.4): menys parametres que aplanar
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    yt, yp = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(imgs)
        yp += out.argmax(1).cpu().tolist()
        yt += labels.tolist()
    return yt, yp


def main():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ----- Dades -----
    Xtr, Ytr, classes = preload(os.path.join(DATA_DIR, "train"), NUM_WORKERS)
    Xva, Yva, _       = preload(os.path.join(DATA_DIR, "val"),   NUM_WORKERS)
    n_classes = len(classes)
    print(f"Classes ({n_classes}): {classes}")
    print(f"Train: {len(Ytr):,}  |  Val: {len(Yva):,}")

    MEAN = Xtr.float().mean().item() / 255.0   # stats reals (millor que 0.5/0.5)
    STD  = Xtr.float().std().item()  / 255.0

    # Hold-out estratificat: el dev surt del TRAIN; el val no es toca fins al final
    idx_tr, idx_dev = train_test_split(
        np.arange(len(Ytr)), test_size=DEV_FRACTION,
        stratify=Ytr.numpy(), random_state=SEED)

    ds_train = MemDS(Xtr, Ytr, MEAN, STD, train=True)
    ds_dev   = MemDS(Xtr, Ytr, MEAN, STD, train=False)
    ds_val   = MemDS(Xva, Yva, MEAN, STD, train=False)

    pin = (device.type == "cuda")
    pw  = NUM_WORKERS > 0
    train_loader = DataLoader(Subset(ds_train, idx_tr), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              persistent_workers=pw, pin_memory=pin)
    dev_loader   = DataLoader(Subset(ds_dev, idx_dev), batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, persistent_workers=pw, pin_memory=pin)
    val_loader   = DataLoader(ds_val, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, persistent_workers=pw, pin_memory=pin)

    # ----- Model i optimitzacio -----
    model = SmallCNN(n_classes, p_drop=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametres entrenables: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Cosine LR: baixa el LR cap al final per afinar la solucio i sortir del
    # plateau. NO surt dels apunts (sec. 14.3 nomes parla d'Adam); afegit.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler(enabled=use_amp)

    # ----- Entrenament -----
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Dev Acc':>7}  {'temps':>7}")
    print("-" * 52)

    best_dev = 0.0
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct = total = 0; loss_sum = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += len(labels)
        scheduler.step()   # un pas per epoca

        train_acc = correct / total
        dev_t, dev_p = predict(model, dev_loader, device, use_amp)
        dev_acc = accuracy_score(dev_t, dev_p)

        print(f"{epoch:>5}  {loss_sum/total:>10.4f}  {train_acc:>8.2%}  "
              f"{dev_acc:>6.2%}  {time.time()-t_start:>5.0f}s")

        if dev_acc > best_dev:           # nomes desem si el dev millora
            best_dev = dev_acc
            torch.save(model.state_dict(), "best_model.pt")

    print(f"\nMillor dev accuracy: {best_dev:.2%}")
    print(f"Temps d'entrenament: {time.time()-t_start:.0f}s")

    # ----- Avaluacio FINAL sobre val (una sola vegada) -----
    model.load_state_dict(torch.load("best_model.pt", map_location=device, weights_only=True))
    val_t, val_p = predict(model, val_loader, device, use_amp)
    micro = f1_score(val_t, val_p, average="micro")   # metrica oficial (= accuracy)
    macro = f1_score(val_t, val_p, average="macro")   # info: classes fluixes
    print(f"\n[VAL] micro-F1: {micro:.4f}  |  macro-F1: {macro:.4f}")


if __name__ == "__main__":
    main()
