#!/bin/sh
# Server llama.cpp per il generatore delle riscritture: 24 slot da 1600 token, flash attention, tutto su GPU.
GGUF=$(ls ~/.cache/huggingface/hub/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/*/Qwen3-4B-Instruct-2507-Q8_0.gguf)
exec "$(dirname "$0")/../tools/llama/llama-server.exe" -m "$GGUF" -ngl 99 -fa on -np 24 -c 38400 -cb \
  --host 127.0.0.1 --port 8090 --no-webui
