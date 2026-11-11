# Pilota del generatore locale (Qwen3-4B-Instruct-2507, transformers, greedy + 2 tentativi a T=0,7)

| prova | problemi | valide | tempo |
|---|---|---|---|
| A, 3 esempi, batch 16 (VRAM satura) | 96 | 96/96 dopo normalizzazione (93 senza) | 228 s |
| C, 3 esempi, batch 16 (VRAM satura) | 96 | 91/96 dopo normalizzazione (58 senza) | 963 s |
| C, 3 esempi, batch 10 | 200 | 183/200 (91,5%) | 367 s |
| C, 1 esempio, batch 20 | 200 | 149/200 (74,5%) | 458 s |

Scarti tipici di C: problemi algebrici con incognite, nomi dentro le espressioni, frazioni come valori.
Collo di bottiglia: pesi 7,5 GB + cache KV ~150 KB/token (prompt ~1300 token) sui 12 GB; oltre ~10,5 GB Windows
riversa in memoria di sistema e un passo di decodifica passa da ~35 ms a >1 s.
