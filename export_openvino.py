import openvino as ov
import torch
import shutil

from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig
from openvino_tokenizers import convert_tokenizer
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
    # LLM
    hf_tokenizer = AutoTokenizer.from_pretrained(pretrained / "LLM")
    llm = OVModelForCausalLM.from_pretrained(pretrained / "LLM", export=True, quantization_config=OVWeightQuantizationConfig(bits=8))
    # llm = OVModelForCausalLM.from_pretrained(pretrained / "LLM", export=True)
    ov_llm_tokenizer, ov_llm_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    llm.save_pretrained(to_dir / "LLM")
    ov.save_model(ov_llm_tokenizer, to_dir / "LLM/openvino_tokenizer.xml")
    ov.save_model(ov_llm_detokenizer, to_dir / "LLM/openvino_detokenizer.xml")

    audio_tokenizer_config = load_config(pretrained / "BiCodec" / "config.yaml")["audio_tokenizer"]

    # MelSpectrogram
    mel_spectrogram = MelSpectrogram(audio_tokenizer_config["mel_params"])

    # Wav2Vec2
    wav2vec = Wav2Vec2Wrapper(pretrained / "wav2vec2-large-xlsr-53")

    # BiCodec
    bicodec = BiCodec.load_from_checkpoint(pretrained / "BiCodec")
    bicodec_tokenizer = BiCodecTokenizer(bicodec)
    bicodec_detokenizer = BiCodecDetokenizer(bicodec)

    # Audio Tokenizer
    example_audio = torch.randn(SAMPLE_RATE * AUDIO_TOKENIZER_DURATION, dtype=torch.float32)  # [96000]
    # Calculate sizes for BiCodecTokenizer
    mel_input = example_audio.unsqueeze(0).unsqueeze(0) # [1, 1, 96000]
    feat_input = example_audio.unsqueeze(0)             # [1, 96000]
    mel = mel_spectrogram(mel_input)                    # [1, 128, 302]
    feat = wav2vec(feat_input)                          # [1, 299, 1024]
    # Convert models to OpenVINO format
    ov_mel_spectrogram = ov.convert_model(mel_spectrogram, example_input=mel_input)
    ov_wav2vec = ov.convert_model(wav2vec, example_input=feat_input)
    ov_bicodec_tokenizer = ov.convert_model(bicodec_tokenizer, example_input=(feat, mel))
    # Save OpenVINO models
    ov.save_model(ov_mel_spectrogram, to_dir / "AudioTokenizer/mel_spectrogram.xml")
    ov.save_model(ov_wav2vec, to_dir / "AudioTokenizer/wav2vec.xml")
    ov.save_model(ov_bicodec_tokenizer, to_dir / "AudioTokenizer/bicodec_tokenizer.xml")


    # # Audio Detokenizer
    # [1, 50], torch.int64
    example_semantic_tokens = torch.randint(0, 1000, (1, 50), dtype=torch.int64)  # Example semantic tokens
    # [1, 1, 32], torch.int32
    example_global_tokens = torch.randint(0, 1000, (1, 1, 32), dtype=torch.int32)  # Example global tokens
    ov_bicodec_detokenizer = ov.convert_model(
        bicodec_detokenizer,
        example_input=(example_semantic_tokens, example_global_tokens)
    )
    ov.save_model(ov_bicodec_detokenizer, to_dir / "AudioDetokenizer/bicodec_detokenizer.xml")

def main():
    pretrained = Path("pretrained_models/Spark-TTS-0.5B")
    to_dir = Path("openvino_models/Spark-TTS-0.5B")

    # remove to_dir if it exists
    if to_dir.exists():
        shutil.rmtree(to_dir)

    export(pretrained, to_dir)

if __name__ == "__main__":
    main()
