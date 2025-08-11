import torch
import shutil
import coremltools as ct
import numpy as np

from pathlib import Path

from openvino_export.export import AudioTokenizer, AudioDetokenizer


AUDIO_TOKENIZER_DURATION = 6  # seconds
AUDIO_DETOKENIZER_DURATION = 1  # seconds
SAMPLE_RATE = 16000  # Hz

def export(pretrained: Path, to_dir: Path):
    # Audio Tokenizer
    example_audio = torch.randn(SAMPLE_RATE * AUDIO_TOKENIZER_DURATION, dtype=torch.float32)  # [96000]

    audio_tokenizer = AudioTokenizer(pretrained)
    audio_tokenizer.eval()
    traced_audio_tokenizer = torch.jit.trace(audio_tokenizer, (example_audio,))

    coreml_audio_tokenizer = ct.convert(
        traced_audio_tokenizer,
        inputs=[ct.TensorType(shape=example_audio.shape, name="audio_input")],
        outputs=[ct.TensorType(name="semantic_tokens"), ct.TensorType(name="global_tokens")],
    )
    coreml_audio_tokenizer.save(to_dir / "AudioTokenizer/AudioTokenizer.mlpackage")

    # Audio Detokenizer
    # [1, 50], torch.int64
    example_semantic_tokens = torch.randint(0, 1000, (1, 50), dtype=torch.int64)  # Example semantic tokens
    # [1, 1, 32], torch.int32
    example_global_tokens = torch.randint(0, 1000, (1, 1, 32), dtype=torch.int32)  # Example global tokens

    audio_detokenizer = AudioDetokenizer(pretrained)
    audio_detokenizer.eval()

    traced_audio_detokenizer = torch.jit.trace(audio_detokenizer, (example_semantic_tokens, example_global_tokens))
    coreml_audio_detokenizer = ct.convert(
        traced_audio_detokenizer,
        inputs=[
            ct.TensorType(shape=example_semantic_tokens.shape, name="semantic_tokens"),
            ct.TensorType(shape=example_global_tokens.shape, name="global_tokens")
        ],
        outputs=[ct.TensorType(name="wav_recon")],
    )

    coreml_audio_detokenizer.save(to_dir / "AudioDetokenizer/AudioDetokenizer.mlpackage")


def main():
    pretrained = Path("pretrained_models/Spark-TTS-0.5B")
    to_dir = Path("coreml_models/Spark-TTS-0.5B")

    # remove to_dir if it exists
    if to_dir.exists():
        shutil.rmtree(to_dir)

    export(pretrained, to_dir)

if __name__ == "__main__":
    main()
