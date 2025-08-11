import torch
import shutil
import coremltools as ct
import numpy as np

from pathlib import Path

from sparktts.models.bicodec import BiCodec
from sparktts.utils.file import load_config

from openvino_export.bicodec import BiCodecTokenizer, BiCodecDetokenizer
from openvino_export.mel_spectrogram import MelSpectrogram
from openvino_export.wav2vec2 import Wav2Vec2Wrapper

AUDIO_TOKENIZER_DURATION = 6  # seconds
AUDIO_DETOKENIZER_DURATION = 1  # seconds
SAMPLE_RATE = 16000  # Hz

def export(pretrained: Path, to_dir: Path):
    audio_tokenizer_config = load_config(pretrained / "BiCodec" / "config.yaml")["audio_tokenizer"]

    # MelSpectrogram
    mel_spectrogram = MelSpectrogram(audio_tokenizer_config["mel_params"])
    mel_spectrogram.eval()

    # Wav2Vec2
    wav2vec = Wav2Vec2Wrapper(pretrained / "wav2vec2-large-xlsr-53")
    wav2vec.eval()

    # BiCodec
    bicodec = BiCodec.load_from_checkpoint(pretrained / "BiCodec")
    bicodec_tokenizer = BiCodecTokenizer(bicodec)
    bicodec_detokenizer = BiCodecDetokenizer(bicodec)
    bicodec_tokenizer.eval()
    bicodec_detokenizer.eval()

    # Audio Tokenizer
    example_audio = torch.randn(SAMPLE_RATE * AUDIO_TOKENIZER_DURATION, dtype=torch.float32)  # [96000]
    # Calculate sizes for BiCodecTokenizer
    mel_input = example_audio.unsqueeze(0).unsqueeze(0) # [1, 1, 96000]
    feat_input = example_audio.unsqueeze(0)             # [1, 96000]
    mel = mel_spectrogram(mel_input)                    # [1, 128, 302]
    feat = wav2vec(feat_input)                          # [1, 299, 1024]

    # Convert models to Core ML format
    traced_mel_spectrogram = torch.jit.trace(mel_spectrogram, (mel_input,))
    traced_wav2vec = torch.jit.trace(wav2vec, (feat_input,))
    traced_bicodec_tokenizer = torch.jit.trace(bicodec_tokenizer, (feat, mel))

    coreml_mel_spectrogram = ct.convert(
        traced_mel_spectrogram,
        inputs=[ct.TensorType(shape=mel_input.shape, name="mel_input")],
        outputs=[ct.TensorType(name="mel_output")],
    )
    coreml_wav2vec = ct.convert(
        traced_wav2vec,
        inputs=[ct.TensorType(shape=feat_input.shape, name="feat_input")],
        outputs=[ct.TensorType(name="feat_output")],
    )
    coreml_bicodec_tokenizer = ct.convert(
        traced_bicodec_tokenizer,
        inputs=[ct.TensorType(shape=feat.shape, name="feat"), ct.TensorType(shape=mel.shape, name="mel")],
        outputs=[ct.TensorType(name="semantic_tokens"), ct.TensorType(name="global_tokens")],
    )

    # Save Core ML models
    coreml_mel_spectrogram.save(to_dir / "AudioTokenizer/MelSpectrogram.mlpackage")
    coreml_wav2vec.save(to_dir / "AudioTokenizer/Wav2Vec.mlpackage")
    coreml_bicodec_tokenizer.save(to_dir / "AudioTokenizer/BiCodecTokenizer.mlpackage")

    # Audio Detokenizer
    # [1, 50], torch.int64
    example_semantic_tokens = torch.randint(0, 1000, (1, 50), dtype=torch.int64)  # Example semantic tokens
    # [1, 1, 32], torch.int32
    example_global_tokens = torch.randint(0, 1000, (1, 1, 32), dtype=torch.int32)  # Example global tokens

    traced_bicodec_detokenizer = torch.jit.trace(bicodec_detokenizer, (example_semantic_tokens, example_global_tokens))
    coreml_bicodec_detokenizer = ct.convert(
        traced_bicodec_detokenizer,
        inputs=[
            ct.TensorType(shape=example_semantic_tokens.shape, name="semantic_tokens"),
            ct.TensorType(shape=example_global_tokens.shape, name="global_tokens")
        ],
        outputs=[ct.TensorType(name="wav_recon")],
    )

    coreml_bicodec_detokenizer.save(to_dir / "AudioDetokenizer/BiCodecDetokenizer.mlpackage")


def main():
    pretrained = Path("pretrained_models/Spark-TTS-0.5B")
    to_dir = Path("coreml_models/Spark-TTS-0.5B")

    # remove to_dir if it exists
    if to_dir.exists():
        shutil.rmtree(to_dir)

    export(pretrained, to_dir)

if __name__ == "__main__":
    main()
