"""GPT v2: loss calcolata solo sulle posizioni con target, senza logits su tutte le posizioni."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model import GPT


class GPT2(GPT):
    def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy media sulle posizioni con targets != -100 (ignore_index=-100)."""
        B, T = idx.shape
        if T > self.ctx_len:
            raise ValueError(f"sequenza di lunghezza {T} oltre ctx_len={self.ctx_len}")
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # lm_head solo dove c'e' un target
        attive = targets != -100
        logits = self.lm_head(x[attive])
        # niente upcast in fp32: i logits restano bf16 (misurato 2x piu' veloce)
        with torch.autocast(device_type=idx.device.type, enabled=False):
            return F.cross_entropy(logits, targets[attive])
