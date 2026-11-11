## Fase K: Qwen3-4B-Base (QLoRA, 1 seed)

**Setup.** QLoRA come in v6: NF4, gradient checkpointing, micro-batch 4, stessi iperparametri LoRA. Sono stati addestrati due adapter nuovi, C5 e A1, in circa 33 minuti ciascuno. Le condizioni 2 (A+calc) e 8 (C6+gram+calc) sono riprese da v6.

**Scelta di A1 per il 4B.** La specifica chiede "il migliore tra A1 e A2 sull'1,7B per accuratezza/token".
- Sull'1,7B A2+calc (0,691, 65 token) è dominato da A2 senza calcolatrice (0,693, 65 token), quindi è fuori dalla frontiera. A1+calc (0,712, 76 token) invece ci sta ed è più accurato di 2 punti (p 0,02).
- Il rapporto accuratezza/token preso alla lettera favorirebbe A2 (0,0106 contro 0,0093), ma premia sempre il formato più corto.
- Ho scelto A1. La scelta è fatta sul test dell'1,7B (vedi Limiti).

| 4B | acc (IC 95%) | token | latenza batch 1 | risposte/s batch 32 | problemi con incognite | frontiera |
|---|---|---|---|---|---|---|
| 2 A+calc (v6) | 0,829 (0,809–0,848) | 102,9 | 5,08 s | 1,14 | 0,590 | sì |
| 5 C5+gram+calc | 0,794 (0,770–0,815) | 75,5 | 3,64 s | 1,72 | 0,393 | sì, per 0,3 token |
| 8 C6+gram+calc (v6) | 0,792 (0,770–0,815) | 81,4 | 3,95 s | 1,52 | 0,443 | no |
| **10 A1+calc** | **0,826** (0,806–0,848) | **75,8** | **3,73 s** | 1,46 | **0,574** | sì |

**Confronti McNemar sul 4B:**

| Confronto | Differenza | p |
|---|---|---|
| A1+calc vs A+calc | −0,2 punti (92 contro 95 discordanti) | 0,88 |
| A1+calc vs C5+gram+calc | +3,3 punti | 0,006 |
| A1+calc vs C6+gram+calc | +3,4 punti | 0,002 |
| C5+gram+calc vs A+calc | −3,5 punti | 0,004 |

**Lettura della frontiera sul 4B:**
- A1+calc ha la stessa accuratezza di A+calc con il 26% di token in meno e il 27% di latenza in meno.
- C5 resta tecnicamente sulla frontiera solo perché usa 0,3 token in meno di A1, con 3,3 punti in meno.
- Sui 61 problemi con incognite A1 (0,574) è quasi pari ad A (0,590), mentre C5 crolla a 0,393.

## Tendenza con la scala

| modello | A+calc | C5+gram+calc | distacco C5 | A1+calc | distacco A1 | token C5 / A | token A1 / A |
|---|---|---|---|---|---|---|---|
| 0,6B | 0,576 | 0,537 | −4,0 | 0,592 | +1,5 | 0,75 | 0,74 |
| 1,7B | 0,725 | 0,691 | −3,4 | 0,712 | −1,3 | 0,73 | 0,75 |
| 4B | 0,829 | 0,794 | −3,5 | 0,826 | −0,2 | 0,73 | 0,74 |

- **C5:** il distacco da A+calc è piatto con la scala, e non si chiude nemmeno a 4B, come per C6 in v6 (−3,6).
- **A1:** a parità di compressione il distacco non ha un segno costante. La media sulle tre scale è circa 0, dentro il rumore (IC per modello di ±2 punti).
- **Conclusione:** la perdita del canonico dipende dalla rappresentazione, non dalla capacità del modello. La prosa concisa invece non paga un costo misurabile.

## Confronto con v5 e v6

- **v5:** C5 perdeva 4,3 e 3,3 punti contro A, per tre cause:
  - output fuori grammatica;
  - mancato recupero dagli errori di conto;
  - ambiguità dei nomi.
- **v6:** grammatica vincolata e calcolatrice hanno eliminato le prime due cause senza chiudere il divario. C6, cioè nomi più espliciti, ha peggiorato.
- **v7:** attacca il problema dall'altro lato. Invece di rendere più ricco un formato simbolico, comprime la prosa e mantiene le parole di relazione. Ottiene la stessa compressione di C5 (−25% di token) senza perdita misurabile contro A, e +2–5,5 punti contro C5.
- **L'analisi degli errori conferma la diagnosi di v6:**
  - nel canonico mancano le parole che dicono come le quantità si relazionano;
  - A1 le conserva, con 0 errori su 20 dovuti al formato;
  - A2 le taglia, con 12 su 20, e il difetto ricompare con la stessa forma degli "nomi che mentono" di C5.

## Limiti e deviazioni

**Deviazioni dalla specifica:**
1. **Selezione morbida dei dati** (vedi Fase I). La copertura stretta è 87% per A1 e 47% per A2, sotto l'obiettivo del 95%. Ho addestrato sul 99,4% e sul 99,6% dei problemi, includendo il tentativo più corto bocciato solo per lunghezza.
   - Di conseguenza A2 comprime il 34% invece del 45% (−36% in uscita contro A+calc).
   - Il 12% dei testi A1 e il 53% dei testi A2 superano la soglia di lunghezza della specifica.
2. **Scelta di A1 per il 4B** per dominanza sulla frontiera, non per il rapporto accuratezza/token letterale.
3. **Un processo per adapter** anche nelle valutazioni, che sono separate per condizione e seed. Sono state aggiunte le opzioni `--cond` e `--seed` a `run_v7.py J`.
4. **Ritentativi del generatore:** il tentativo bocciato solo per lunghezza riceve una richiesta esplicita di accorciare con il numero di token. È un'aggiunta rispetto al semplice "ritenta".

**Limiti:**
- **Seed:** 1 sul 4B e 2 sull'1,7B. I confronti A1 contro A sull'1,7B e sul 4B non sono significativi in nessuna direzione, quindi "pari" vuol dire "differenza < ~2 punti", non "identici".
- **Scelta tra A1 e A2:** è stata fatta sui risultati di test dell'1,7B, non su un dev set. È una lieve contaminazione della selezione, irrilevante per il confronto con C5, che non dipende dalla scelta.
- **Condizioni diverse tra formati:** A1 e A2 sono valutati senza grammatica, solo con calcolatrice, mentre C5 ha anche la grammatica. In v6 la grammatica costava a C5 circa 1 punto. Il confronto senza grammatica né calcolatrice (A1 contro C5, condizioni 9 e 3) dà comunque +5,3 punti sullo 0,6B e +1,4 sull’1,7B; sul 4B non è stato misurato.
- **Deriva della latenza:** latenze e throughput delle condizioni nuove sono stati misurati in una sessione diversa da quelle riprese da v6. Sull'1,7B i ms per token sono il ~10% più alti (18,2–18,6 contro 16,6–16,7), quindi i guadagni di latenza dell'1,7B sono sottostimati. Il conteggio dei token è la misura più affidabile.
- **Origine dei dati:** le riscritture vengono da Qwen3-4B-Instruct, la stessa famiglia dei modelli addestrati, e partono da A. Ereditano quindi i passaggi di A e la sua scelta dei conti.
- **Analisi degli errori:** 20 errori per condizione, un solo classificatore LLM, solo seed 0. I conteggi sono indicativi.
- **Recupero:** la colonna `recovery` nelle condizioni con calcolatrice si basa su pochissimi casi (0,2–1% di uguaglianze sbagliate) e non è interpretabile. Il recupero si legge solo nelle condizioni senza calcolatrice (1, 3, 9 e 11).

## File

| file | contenuto |
|---|---|
| `results/v7/runs.csv` | una riga per run (modello, condizione, seed) |
| `results/v7/latency.csv` | latenze dei run |
| `results/v7/summary.csv` | medie per condizione con IC, recupero, incognite e frontiera |
| `results/v7/comparisons.csv` | McNemar aggregato e per seed |
| `results/v7/scale.csv` | tendenza con la scala |
| `results/v7/plots/` | frontiera accuratezza/token e accuratezza/latenza per modello |
| `results/v7/data_a7.json`, `dropped_A{1,2}.csv`, `examples_A_A1_A2_C5.md` | dati della Fase I |
| `results/v7/a_calc_audit_A{1,2}.json` | audit della calcolatrice |
| `results/v7/errors_sample.jsonl`, `errors_classified.csv` | analisi degli errori |
| `data/rewrites/A{1,2}.jsonl` | tutti i tentativi del generatore |
| `data/sft/A{1,2}_train.jsonl` | dati di training |
| `results/v7/outputs/` | output per problema di ogni run |
