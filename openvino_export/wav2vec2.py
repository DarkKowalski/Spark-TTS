import torch
from torch import nn
from transformers import Wav2Vec2Model
from pathlib import Path

class Wav2Vec2Wrapper(nn.Module):
    def __init__(self, pretrained_dir: str, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.model = Wav2Vec2Model.from_pretrained(pretrained_dir).to(device)
        self.model.config.output_hidden_states = True

    # input shape [1, samples]
    def forward(self, mono_audio: torch.Tensor) -> torch.Tensor:
        feat = self.model(mono_audio.to(self.device))
        feats_mix = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        return feats_mix
