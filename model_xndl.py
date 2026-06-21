# Laboratori XNDL final notat - classificacio d'imatges 32x32 en 14 categories
#
# L'esquelet del professorat usava una xarxa fully-connected: aplanava la
# imatge i la passava per capes denses. Aixo tracta cada pixel com a
# independent i ignora l'estructura espacial, a mes de gastar molts parametres.
# El substituim per una CNN, que comparteix els filtres per tota la imatge
# (detecta el mateix patro en qualsevol posicio) amb molts menys parametres.
#
# Metodologia (sense data leakage):
#   - El 'val' nomes s'usa per a l'avaluacio FINAL.
#   - La seleccio de model es fa amb un hold-out estratificat tret del 'train'
#     (dev), i nomes desem el model quan el dev millora.

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
DATA_DIR        = os.getenv("DATA_DIR", "./dataset_npz")
BATCH_SIZE      = 256          # batch gran: gradient estable i bon aprofitament de la GPU
EPOCHS          = 30
LR              = 1e-3         # learning rate inicial per a Adam
WEIGHT_DECAY    = 1e-4         # penalitzacio L2 suau (el gap train/dev no mostra overfitting)
DROPOUT         = 0.2          # apaga neurones al classificador per evitar coadaptacio
DEV_FRACTION    = 0.10         # part del train reservada per seleccionar el millor model
NUM_WORKERS     = 4            # processos que preparen dades en paral.lel. A Windows, si peta, posa 0
SEED            = 42


# Ja no s'usa el NpzImageDataset
'''
class NpzImageDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.images = data['images']
        self.labels = data['labels']
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img).unsqueeze(0)
        label = int(self.labels[idx])
        return img, label

# I ara instancies els datasets així:
train_ds = NpzImageDataset(os.path.join(DATADIR, 'train.npz'))
val_ds = NpzImageDataset(os.path.join(DATADIR, 'val.npz'))
'''

# Carreguem cada split sencer a memoria (uint8) un sol cop. Llegir 112k PNGs
# del disc a cada epoca era el coll d'ampolla real; fent-ho una vegada, les
# epoques deixen de dependre del disc i en caben moltes mes dins del temps.
# Versió antiga amb ImageFolder
'''
def preload(folder, workers):
    ds = datasets.ImageFolder(folder, transform=transforms.Compose([
        transforms.Grayscale(), transforms.PILToTensor()]))  # uint8 [1,32,32]
    loader = DataLoader(ds, batch_size=1024, num_workers=workers)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)
    return torch.cat(xs), torch.cat(ys), ds.classes
'''

# Nova versió

def preload_npz(npz_path):
    data = np.load(npz_path)
    X = torch.from_numpy(data['images'])
    Y = torch.from_numpy(data['labels']).long()
    if X.ndim == 3:
        X = X.unsqueeze(1)
    return X, Y


class MemDS(Dataset):
    def __init__(self, X, Y, mean, std, train):
        self.X, self.Y = X, Y
        aug = []
        if train:
            # Augmentacio nomes al train: genera variacions realistes perque el
            # model generalitzi millor. Crop amb padding + flip es barat (no fa
            # interpolacio) i dona robustesa a petits desplacaments i al mirall.
            aug = [transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(0.5)]
        # Passem a float [0,1] i normalitzem amb la mitjana/std reals del train
        # (centra les dades i estabilitza l'entrenament millor que 0.5/0.5).
        self.tf = transforms.Compose(
            [transforms.ConvertImageDtype(torch.float32)] + aug +
            [transforms.Normalize((mean,), (std,))])

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.tf(self.X[i]), int(self.Y[i])


# CNN amb forma de piramide: la mida espacial baixa (pooling) mentre el nombre
# de canals puja, de manera que les primeres capes capten patrons simples
# (vores, traços) i les profundes els combinen en formes completes.
class SmallCNN(nn.Module):
    def __init__(self, n_classes, in_ch=1, p_drop=0.2):
        super().__init__()

        def block(c_in, c_out):
            # Tres convolucions 3x3 apilades per bloc: cada una afegeix una
            # no-linealitat i amplia el camp receptiu efectiu, de manera que el
            # bloc capta patrons mes complexos amb molts menys parametres que
            # un sol filtre gran. Ampliem aixi la profunditat del model donat,
            # que es el que ens faltava (anavem justos de capacitat). El
            # max-pool final redueix la mida i aporta invariancia a petites
            # translacions.
            return nn.Sequential(
                nn.Conv2d(c_in,  c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),   # BatchNorm: normalitza activacions i estabilitza
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, 64),    # 32 -> 16
            block(64, 128),      # 16 -> 8
            block(128, 256),     # 8  -> 4
        )
        self.head = nn.Sequential(
            # Global average pooling en lloc d'aplanar + capa densa gran: molts
            # menys parametres i menys risc de sobreajustament.
            nn.AdaptiveAvgPool2d(1),
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
    torch.backends.cudnn.benchmark = True   # mida d'entrada fixa -> cuDNN tria el millor algorisme
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ----- Dades -----
    Xtr, Ytr = preload_npz(os.path.join(DATA_DIR, "train.npz"))
    Xva, Yva = preload_npz(os.path.join(DATA_DIR, "val.npz"))

    with open(os.path.join(DATA_DIR, "classes.txt"), "r") as f:
        classes = [line.strip() for line in f if line.strip()]

    n_classes = len(classes)
    print(f"Classes ({n_classes}): {classes}")
    print(f"Train: {len(Ytr):,}  |  Val: {len(Yva):,}")

    MEAN = Xtr.float().mean().item() / 255.0
    STD  = Xtr.float().std().item()  / 255.0

    # Partim el train en train-real + dev (estratificat per mantenir la
    # proporcio de classes). El dev guia la seleccio de model; el val queda
    # intacte fins al final per donar una estimacio neta, sense leakage.
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
    # Baixem el learning rate de forma progressiva (cosinus) al llarg de
    # l'entrenament: passos grans al principi per avancar de pressa i passos
    # petits al final per afinar la solucio i sortir del plateau.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    use_amp = (device.type == "cuda")   # mixed precision: mes rapid a la GPU, sense perdre qualitat
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
        scheduler.step()

        train_acc = correct / total
        dev_t, dev_p = predict(model, dev_loader, device, use_amp)
        dev_acc = accuracy_score(dev_t, dev_p)

        print(f"{epoch:>5}  {loss_sum/total:>10.4f}  {train_acc:>8.2%}  "
              f"{dev_acc:>6.2%}  {time.time()-t_start:>5.0f}s")

        # Ens quedem nomes amb el millor model segons el dev (descartem les
        # epoques que empitjoren).
        if dev_acc > best_dev:
            best_dev = dev_acc
            torch.save(model.state_dict(), "best_model.pt")

    print(f"\nMillor dev accuracy: {best_dev:.2%}")
    print(f"Temps d'entrenament: {time.time()-t_start:.0f}s")

    # ----- Avaluacio FINAL sobre val (nomes ara, un sol cop) -----
    model.load_state_dict(torch.load("best_model.pt", map_location=device, weights_only=True))
    val_t, val_p = predict(model, val_loader, device, use_amp)
    micro = f1_score(val_t, val_p, average="micro")   # metrica oficial demanada (= accuracy)
    macro = f1_score(val_t, val_p, average="macro")   # informatiu: detecta classes fluixes
    print(f"\n[VAL] micro-F1: {micro:.4f}  |  macro-F1: {macro:.4f}")


if __name__ == "__main__":
    main()
