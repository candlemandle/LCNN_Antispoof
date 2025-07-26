import os
import torch
import librosa
from torch.utils.data import Dataset


target_map = {'bonafide': 0, 'spoof': 1}

class ASVDataset(Dataset):
    def __init__(self, root_dir: str, subset: str, feature_extractor):

        self.audio_dir = os.path.join(root_dir, subset, 'flac')

        self.samples = self._load_list(root_dir, subset)
        self.fe      = feature_extractor

    def _load_list(self, root_dir: str, subset: str):

        proto_path = os.path.join(root_dir, subset, f"{subset}.txt")
        lines      = open(proto_path, 'r').read().splitlines()
        samples    = []
        for line in lines:
            parts   = line.split()
            file_id = parts[1]
            label   = parts[-1]
            samples.append((file_id, target_map[label]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_id, label = self.samples[idx]

        wav_path = os.path.join(self.audio_dir, file_id + '.flac')

        wav_np, sr = librosa.load(wav_path, sr=None, mono=True)

        wav = torch.from_numpy(wav_np).unsqueeze(0)

        spec = self.fe(wav)
        return spec, torch.tensor(label, dtype=torch.long)
