import torch
import shutil
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

    # To ONNX
    onnx_mel_spectrogram = torch.onnx.export(mel_spectrogram, (mel_input,), dynamo=True)
    onnx_wav2vec = torch.onnx.export(wav2vec, (feat_input,), dynamo=True)
    onnx_bicodec_tokenizer = torch.onnx.export(bicodec_tokenizer, (feat, mel), dynamo=True)

    # Save ONNX models
    onnx_mel_spectrogram.save(to_dir / "AudioTokenizer/mel_spectrogram.onnx")
    onnx_wav2vec.save(to_dir / "AudioTokenizer/wav2vec.onnx")
    onnx_bicodec_tokenizer.save(to_dir / "AudioTokenizer/bicodec_tokenizer.onnx")

    # Audio Detokenizer
    # [1, 50], torch.int64
    example_semantic_tokens = torch.randint(0, 1000, (1, 50), dtype=torch.int64)  # Example semantic tokens
    # [1, 1, 32], torch.int32
    example_global_tokens = torch.randint(0, 1000, (1, 1, 32), dtype=torch.int32)  # Example global tokens

    onnx_bicodec_detokenizer = torch.onnx.export(
        bicodec_detokenizer,
        (example_semantic_tokens, example_global_tokens),
        dynamo=True
    )
    onnx_bicodec_detokenizer.save(to_dir / "AudioDetokenizer/bicodec_detokenizer.onnx")


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
