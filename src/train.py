import argparse
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.features import SpectrogramExtractor
from src.utils    import ASVDataset
from src.model    import LCNN
from src.calculate_eer import compute_eer


def collate_fn(batch):
    specs, labels = zip(*batch)
    max_frames = max(s.shape[-1] for s in specs)
    padded = [F.pad(s, (0, max_frames - s.shape[-1])) for s in specs]
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


def train(cfg):
    device = torch.device(
        'mps'  if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )


    cfg['lr']         = float(cfg['lr'])
    cfg['batch_size'] = int(cfg['batch_size'])
    cfg['epochs']     = int(cfg['epochs'])


    fe       = SpectrogramExtractor(**cfg['fe'])
    train_ds = ASVDataset(cfg['data_root'], 'train', fe)
    dev_ds   = ASVDataset(cfg['data_root'], 'dev',   fe)

    num_workers = 1 if device.type=='mps' else 2
    train_ld = DataLoader(train_ds,
                          batch_size=cfg['batch_size'],
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=num_workers,
                          pin_memory=(device.type!='cpu'))
    dev_ld   = DataLoader(dev_ds,
                          batch_size=cfg['batch_size'],
                          shuffle=False,
                          collate_fn=collate_fn,
                          num_workers=num_workers,
                          pin_memory=(device.type!='cpu'))

    model     = LCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])


    metrics_file = cfg.get('metrics_path','metrics.csv')
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','dev_eer'])

    best_eer = float('inf')
    print(f"\nUsing device: {device}\n")


    for epoch in tqdm(range(1, cfg['epochs']+1), desc="Epochs", unit="ep"):

        model.train()
        total_loss = 0.0
        for i, (x, y) in enumerate(train_ld, 1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"  [E{epoch}] batch {i}/{len(train_ld)} — loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_ld)


        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for x, y in dev_ld:
                x = x.to(device)
                logits = model(x)
                probs  = F.softmax(logits, dim=1)[:,1].cpu().numpy()
                all_scores.extend(probs.tolist())
                all_labels.extend(y.numpy().tolist())

        eer, _ = compute_eer(all_labels, all_scores)
        print(f"  → Epoch {epoch} complete: Train Loss={avg_loss:.4f}, Dev EER={eer:.4f}\n")


        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), cfg.get('ckpt_path','best.ckpt'))


        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_loss:.4f}", f"{eer:.4f}"])

    print(f"Training done. Best Dev EER={best_eer:.4f}")
    print(f"Metrics logged to {metrics_file}, best model at {cfg.get('ckpt_path','best.ckpt')}")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train(cfg)