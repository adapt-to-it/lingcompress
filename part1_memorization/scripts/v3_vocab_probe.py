"""Stima per la Fase D: token/bio per formato e vocabolario (BPE con cifre singole) e alfabeto char."""
import pandas as pd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from src.config import EOS, PAD
from src.v2 import corpus
from src.v2.spec import CORPORA
persons = corpus.load_persons(20000)
rows = []
for fmt in ("italiano_pieno", "telegrafico", "interlingua_v1enc"):
    testi = corpus.epoch_texts(CORPORA[fmt], persons, 0)
    alfabeto = sorted(set("".join(testi)))
    lens = [len(t) for t in testi]
    rows.append(dict(format=fmt, vocab="char", vocab_eff=len(alfabeto) + 2, tok_bio=sum(lens) / len(lens), max_len=max(lens) + 2))
    for v in (512, 1024, 2048, 4096):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel(add_prefix_space=False)])
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(testi, trainers.BpeTrainer(vocab_size=v, special_tokens=[EOS, PAD], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        enc = [len(e.ids) for e in tok.encode_batch(testi)]
        rows.append(dict(format=fmt, vocab=v, vocab_eff=tok.get_vocab_size(), tok_bio=sum(enc) / len(enc), max_len=max(enc) + 2))
    print(rows[-5:], flush=True)
df = pd.DataFrame(rows); print(df.round(1).to_string()); df.to_csv("results/v3/vocab_probe.csv", index=False)
