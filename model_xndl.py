# Laboratori XNDL final notat - classificacio d'imatges 32x32 en 14 categories
#
# L'esquelet proporcionat usava una xarxa fully-connected: aplanava la
# imatge i la passava per capes denses. Aixo tracta cada pixel com a
# independent i ignora l'estructura espacial, a mes de gastar molts parametres.
# El substituim per una CNN, que comparteix els filtres per tota la imatge
# (detecta el mateix patro en qualsevol posicio) amb molts menys parametres.
#
# Hem seguit la següent metodologia per tal d'evitar tenir data leakage:
#   - El 'val' nomes s'usa per a l'avaluacio FINAL.
#   - La seleccio de model es fa amb un hold-out estratificat tret del 'train'
#     (dev), i nomes desem el model quan el dev millora.

import os, time, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# Hiperparametres
DATA_DIR        = os.getenv("DATA_DIR", "./dataset_npz")
BATCH_SIZE      = 256
LR              = 1e-3
WEIGHT_DECAY    = 6e-4
DROPOUT         = 0.3
DEV_FRACTION    = 0.10
NUM_WORKERS     = 4
SEED            = 42
LABEL_SMOOTHING = 0.05

# El temps mana; les epoques nomes son un sostre de seguretat.
TIME_LIMIT      = 1150
EPOCHS          = 250


# Carreguem el dataset
def preload_npz(npz_path):
    data = np.load(npz_path)
    X = torch.from_numpy(data["images"])          
    Y = torch.from_numpy(data["labels"]).long()
    if X.ndim == 3:                               
        X = X.unsqueeze(1)
    return X, Y


class MemDS(Dataset):
    def __init__(self, X, Y, mean, std, train):
        self.X, self.Y = X, Y
        aug = []
        if train:
            # Augmentacio nomes al train: genera variacions realistes perque el
            # model generalitzi millor i no memoritzi. Rotacio + desplacament +
            # escala + mirall; aquestes categories segueixen sent la mateixa
            # classe sota aquestes transformacions. Ens serveix per controlar l'overfitting
            # (fa el train mes variat enlloc de limitar el model).
            aug = [
                transforms.RandomAffine(
                    degrees=12,
                    translate=(0.08, 0.08),
                    scale=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(0.5)
            ]

        # Passem a float [0,1] i normalitzem amb la mitjana/std reals del train
        self.tf = transforms.Compose(
            [transforms.ConvertImageDtype(torch.float32)] + aug +
            [transforms.Normalize((mean,), (std,))]
        )

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.tf(self.X[i]), int(self.Y[i])


# CNN amb forma de piramide: la mida espacial baixa (pooling) mentre el nombre
# de canals puja, de manera que les primeres capes capten patrons simples
# i les profundes els combinen en formes completes.
class SmallCNN(nn.Module):
    def __init__(self, n_classes, in_ch=1, p_drop=0.3):
        super().__init__()

        def block(c_in, c_out):
            # Tres convolucions 3x3 apilades per bloc: cada una afegeix una
            # no-linealitat i amplia el camp receptiu efectiu, de manera que el
            # bloc capta patrons mes complexos amb molts menys parametres que
            # un sol filtre gran.
            # El max-pool final redueix la mida i aporta invariancia a petites translacions.
            return nn.Sequential(
                nn.Conv2d(c_in,  c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
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
            # Global average pooling en lloc d'aplanar + capa densa gran: genera molts
            # menys parametres i menys risc de sobreajustament.
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # Prediccio normal sobre la imatge original
        out = self.head(self.features(x))

        # Durant l'entrenament NO fem el truc del mirall.
        # Aixi el train segueix sent normal i no dupliquem el forward.
        if self.training:
            return out

        # En avaluacio/inferencia, el professor fara model(x).
        # Per tant, el truc del mirall ha d'estar dins del forward.
        out_flip = self.head(self.features(torch.flip(x, dims=[3])))

        # Mitjana dels logits de la imatge original i de la imatge reflectida.
        return (out + out_flip) / 2


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

    # Dades
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
    # intacte fins al final per donar una estimacio neta, sense fuites d'informació.
    idx_tr, idx_dev = train_test_split(
        np.arange(len(Ytr)),
        test_size=DEV_FRACTION,
        stratify=Ytr.numpy(),
        random_state=SEED
    )

    ds_train = MemDS(Xtr, Ytr, MEAN, STD, train=True)
    ds_dev   = MemDS(Xtr, Ytr, MEAN, STD, train=False)
    ds_val   = MemDS(Xva, Yva, MEAN, STD, train=False)

    pin = (device.type == "cuda")
    pw  = NUM_WORKERS > 0

    train_loader = DataLoader(
        Subset(ds_train, idx_tr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=pw,
        pin_memory=pin
    )

    dev_loader = DataLoader(
        Subset(ds_dev, idx_dev),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=pw,
        pin_memory=pin
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=pw,
        pin_memory=pin
    )

    # Model i optimitzacio
    model = SmallCNN(n_classes, p_drop=DROPOUT).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametres entrenables: {n_params:,}")

    # Canvi afegit: label smoothing per reduir sobreconfiança del model
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    use_amp = (device.type == "cuda")   # mixed precision: mes rapid a la GPU
    scaler  = torch.amp.GradScaler(enabled=use_amp)

    # Entrenament
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Dev Acc':>7}  {'lr':>8}  {'temps':>7}")
    print("-" * 62)

    best_dev = 0.0
    t_start = time.time()
    last_ep = 0.0

    for epoch in range(1, EPOCHS + 1):
        elapsed = time.time() - t_start

        # Aturem si no hi cap una epoca mes dins del limit (mai a mitja epoca).
        if epoch > 1 and elapsed + last_ep > TIME_LIMIT:
            print("Temps esgotat; aturant per cabre en el limit.")
            break

        # Cosine basat en temps: baixa el LR de LR fins a aprox 0 al llarg de
        # TIME_LIMIT, independentment de quantes epoques surtin.
        frac = min(elapsed / TIME_LIMIT, 1.0)
        cur_lr = 0.5 * LR * (1.0 + math.cos(math.pi * frac))

        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        ep_t0 = time.time()

        model.train()
        correct = total = 0
        loss_sum = 0.0

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

        last_ep = time.time() - ep_t0

        train_acc = correct / total

        dev_t, dev_p = predict(model, dev_loader, device, use_amp)
        dev_acc = accuracy_score(dev_t, dev_p)

        print(
            f"{epoch:>5}  {loss_sum/total:>10.4f}  {train_acc:>8.2%}  "
            f"{dev_acc:>6.2%}  {cur_lr:>8.5f}  {time.time()-t_start:>5.0f}s"
        )

        # Ens quedem nomes amb el millor model segons el dev (descartem les
        # epoques que empitjoren).
        if dev_acc > best_dev:
            best_dev = dev_acc
            torch.save(model.state_dict(), "best_model.pt")

    print(f"\nMillor dev accuracy: {best_dev:.2%}")
    print(f"Temps d'entrenament: {time.time()-t_start:.0f}s")

    # Un cop finalitzat l'entrenament avaluem el millor model sobre val
    model.load_state_dict(
        torch.load("best_model.pt", map_location=device, weights_only=True)
    )

    val_t, val_p = predict(model, val_loader, device, use_amp)

    micro = f1_score(val_t, val_p, average="micro")
    macro = f1_score(val_t, val_p, average="macro")

    print(f"\n[VAL] micro-F1: {micro:.4f}  |  macro-F1: {macro:.4f}")


if __name__ == "__main__":
    main()