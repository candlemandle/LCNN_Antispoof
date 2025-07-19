import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.features import SpectrogramExtractor
from src.utils    import ASVDataset
from src.model    import LCNN
from src.calculate_eer import compute_eer

def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # feature extractor & datasets
    fe = SpectrogramExtractor(**cfg['fe'])
    train_ds = ASVDataset(cfg['data_root'], 'train', fe)
    dev_ds   = ASVDataset(cfg['data_root'], 'dev',   fe)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg['batch_size'], shuffle=False)

    # model + losses and optimizer
    model     = LCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    best_eer = 1.0
    for epoch in range(1, cfg['epochs']+1):
        # training
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for x, y in dev_loader:
                x = x.to(device)
                logits = model(x)
                probs  = F.softmax(logits, dim=1)[:,1].cpu().numpy()
                all_scores.extend(probs.tolist())
                all_labels.extend(y.numpy().tolist())
        eer, _ = compute_eer(all_labels, all_scores)

        print(f"Epoch {epoch}/{cfg['epochs']} — Train Loss: {avg_loss:.4f} — Dev EER: {eer:.4f}")

        # saving the best
        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), cfg.get('ckpt_path','best.ckpt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train(cfg)
