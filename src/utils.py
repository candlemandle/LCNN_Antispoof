import os
import torch
import librosa
from torch.utils.data import Dataset

target_map = {'bonafide': 0, 'spoof': 1}

class ASVDataset(Dataset):
    def __init__(self, root_dir, subset, feature_extractor):
        self.root = os.path.join(root_dir, subset, 'flac')
        self.lst = self._load_list(subset)
        self.fe = feature_extractor

    def _load_list(self, subset):
        key_file = os.path.join(self.root, '..', subset + '.txt')
        lines = open(key_file).read().splitlines()
        return [(os.path.join(self.root, utt + '.flac'), target_map[label])
                for utt, label in (line.split() for line in lines)]

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        path, label = self.lst[idx]
        wav_np, sr = librosa.load(path, sr=None, mono=True)
        wav = torch.from_numpy(wav_np).unsqueeze(0)  # (1, time)
        spec = self.fe(wav)
        return spec, torch.tensor(label, dtype=torch.long)
