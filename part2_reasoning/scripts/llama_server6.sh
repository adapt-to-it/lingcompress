#!/bin/sh
# Server llama.cpp per C6: prompt piu' lungo (4 esempi) -> 24 slot da 2048 token con KV cache q8_0.
GGUF=$(ls ~/.cache/huggingface/hub/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/*/Qwen3-4B-Instruct-2507-Q8_0.gguf)
exec "$(dirname "$0")/../tools/llama/llama-server.exe" -m "$GGUF" -ngl 99 -fa on -np 24 -c 49152 -cb \
  -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port 8090 --no-webui
