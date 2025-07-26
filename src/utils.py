import os
import torch
from torch.utils.data import Dataset

target_map = {'bonafide': 0, 'spoof': 1}

class ASVDataset(Dataset):
    def __init__(self, root_dir: str, subset: str, feature_extractor=None):
        self.spec_dir = os.path.join(root_dir, 'spec', subset)
        proto_path    = os.path.join(root_dir, subset, f"{subset}.txt")
        lines         = open(proto_path, 'r').read().splitlines()
        self.samples  = []
        for line in lines:
            parts   = line.split()
            file_id = parts[1]
            label   = parts[-1]
            self.samples.append((file_id, target_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_id, label = self.samples[idx]
        spec_path      = os.path.join(self.spec_dir, file_id + '.pt')
        spec           = torch.load(spec_path)
        return spec, torch.tensor(label, dtype=torch.long)
