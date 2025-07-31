#!/bin/bash -eu

docker run -it --rm -v ./pretrained_models:/models -v ./gguf:/gguf ghcr.io/ggml-org/llama.cpp:full --convert --outtype f32 --outfile /gguf/ "/models/Spark-TTS-0.5B/LLM"
docker run -it --rm -v ./gguf:/gguf ghcr.io/ggml-org/llama.cpp:full --quantize /gguf/LLM-507M-F32.gguf /gguf/LLM-507M-Q4_K.gguf 15
