from openvino_export.bicodec import BiCodecTokenizer, BiCodecDetokenizer
from openvino_export.mel_spectrogram import MelSpectrogram
from openvino_export.wav2vec2 import Wav2Vec2Wrapper

from pathlib import Path

from sparktts.models.bicodec import BiCodec
from sparktts.utils.file import load_config

import torch
from torch import nn

class AudioTokenizer(nn.Module):
    def __init__(self, pretrained: Path):
        super().__init__()
        audio_tokenizer_config = load_config(pretrained / "BiCodec" / "config.yaml")["audio_tokenizer"]

        mel_params = audio_tokenizer_config["mel_params"]

        # MelSpectrogram
        self.mel_spectrogram = MelSpectrogram(mel_params)
        self.mel_spectrogram.eval()

        # Wav2Vec2
        self.wav2vec = Wav2Vec2Wrapper(pretrained / "wav2vec2-large-xlsr-53")
        self.wav2vec.eval()

        # BiCodec
        bicodec = BiCodec.load_from_checkpoint(pretrained / "BiCodec")
        self.bicodec_tokenizer = BiCodecTokenizer(bicodec)
        self.bicodec_tokenizer.eval()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mel_input = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, 96000]
        feat_input = audio.unsqueeze(0)              # [1, 96000]
        
        mel = self.mel_spectrogram(mel_input)        # [1, 128, 302]
        feat = self.wav2vec(feat_input)              # [1, 299, 1024]

        return self.bicodec_tokenizer(feat, mel)     # Returns semantic and global tokens

class AudioDetokenizer(nn.Module):
    def __init__(self, pretrained: Path):
        super().__init__()
        # Load BiCodec model
        bicodec = BiCodec.load_from_checkpoint(pretrained / "BiCodec")
        self.bicodec_detokenizer = BiCodecDetokenizer(bicodec)
        self.bicodec_detokenizer.eval()

    def forward(self, semantic_tokens: torch.Tensor, global_tokens: torch.Tensor) -> torch.Tensor:
        return self.bicodec_detokenizer(semantic_tokens, global_tokens)
