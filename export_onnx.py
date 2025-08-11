import torch
import shutil
from pathlib import Path

from sparktts.models.bicodec import BiCodec
from sparktts.utils.file import load_config

from openvino_export.export import AudioTokenizer, AudioDetokenizer

AUDIO_TOKENIZER_DURATION = 6  # seconds
AUDIO_DETOKENIZER_DURATION = 1  # seconds
SAMPLE_RATE = 16000  # Hz
OPSET_VERSION = 14  # ONNX opset version

def export(pretrained: Path, to_dir: Path):
    # Audio Tokenizer
    example_audio = torch.randn(SAMPLE_RATE * AUDIO_TOKENIZER_DURATION, dtype=torch.float32)  # [96000]

    audio_tokenizer = AudioTokenizer(pretrained)
    audio_tokenizer.eval()

    # To ONNX
    torch.onnx.export(
        audio_tokenizer,
        (example_audio,),
        to_dir / "AudioTokenizer/AudioTokenizer.onnx",
        input_names=["audio_input"], 
        output_names=["semantic_tokens", "global_tokens"],
        opset_version=OPSET_VERSION
    )
    
    # Audio Detokenizer
    # [1, 50], torch.int64
    example_semantic_tokens = torch.randint(0, 1000, (1, 50), dtype=torch.int64)  # Example semantic tokens
    # [1, 1, 32], torch.int32
    example_global_tokens = torch.randint(0, 1000, (1, 1, 32), dtype=torch.int32)  # Example global tokens

    audio_detokenizer = AudioDetokenizer(pretrained)
    audio_detokenizer.eval()

    torch.onnx.export(
        audio_detokenizer,
        (example_semantic_tokens, example_global_tokens),
        to_dir / "AudioDetokenizer/AudioDetokenizer.onnx",
        input_names=['semantic_tokens', 'global_tokens'],
        output_names=['wav_recon'],
        opset_version=OPSET_VERSION
    )

def main():
    pretrained = Path("pretrained_models/Spark-TTS-0.5B")
    to_dir = Path("onnx_models/Spark-TTS-0.5B")

    # remove to_dir if it exists
    if to_dir.exists():
        shutil.rmtree(to_dir)
    to_dir.mkdir(parents=True, exist_ok=True)
    (to_dir / "AudioTokenizer").mkdir(parents=True, exist_ok=True)
    (to_dir / "AudioDetokenizer").mkdir(parents=True, exist_ok=True)

    export(pretrained, to_dir)

if __name__ == "__main__":
    main()
