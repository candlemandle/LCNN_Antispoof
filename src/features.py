import torch

class SpectrogramExtractor:
    def __init__(self,
                 n_fft=512,
                 hop_length=128,
                 win_length=512,
                 power=2.0):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.power = power

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length,
                                     device=waveform.device),
            return_complex=True
        )
        spec = torch.abs(spec) ** self.power
        return spec.unsqueeze(0)  # shape: (1, 1, freq_bins, frames)
