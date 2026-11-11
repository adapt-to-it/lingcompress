Il codice è in `v5/`: orchestratore `v5/run_v7.py`, moduli nuovi `src/spec7.py`, `src/concise7.py` e `src/analysis7.py`, script `scripts/run_j7.sh` e `scripts/run_k7.sh`. I risultati sono in `results/v7/`, i grafici della frontiera in `results/v7/plots/frontier_{small,large,xl}.png`. Le condizioni 1, 2, 3 e 5 (0,6B e 1,7B) e 2 e 8 (4B) sono riprese da v6.

## Risposta in breve: la prosa concisa batte il canonico sulla frontiera?

**Sì.** La prosa concisa A1 con la calcolatrice (condizione 10) domina il miglior canonico, C5 con grammatica e calcolatrice (condizione 5), a tutte e tre le scale. Usa gli stessi token (circa il 74–75% di A) ed è più accurata di 5,5, 2,1 e 3,3 punti. In più raggiunge A con calcolatrice: +1,5 punti, −1,3 (non significativo) e −0,2 (non significativo).

| | A+calc (2) | C5+gram+calc (5) | **A1+calc (10)** | A2+calc (12) |
|---|---|---|---|---|
| 0,6B (3 seed) | 57,6% · 101 tok | 53,7% · 76 tok | **59,2% · 75 tok** | 56,0% · 64 tok |
| 1,7B (2 seed) | 72,5% · 102 tok | 69,1% · 74 tok | **71,2% · 76 tok** | 69,1% · 65 tok |
| 4B QLoRA (1 seed) | 82,9% · 103 tok | 79,4% · 76 tok | **82,6% · 76 tok** | – |

- **A1 contro C5 + gram + calc:**
  - 0,6B: +5,5 punti, p < 10⁻¹¹, p < 0,001 in ogni seed.
  - 1,7B: +2,1 punti, p 0,03.
  - 4B: +3,3 punti, p 0,006.
  - I token sono gli stessi: A1 ha 0,8 token in meno, 2,4 in più e 0,3 in più. Sull'1,7B C5 è dominato anche da A1 senza calcolatrice e da A2.
- **A1 contro A + calc:**
  - 0,6B: +1,5 punti, p 0,04 (non stabile tra i seed).
  - 1,7B: −1,3 punti, p 0,12.
  - 4B: −0,2 punti, p 0,88.
  - Il risparmio di token è del 25–26% e la latenza a batch 1 scende del 14–27%.
  - Sullo 0,6B A1+calc domina A+calc; sull'1,7B e sul 4B lo affianca sulla frontiera, con una perdita non significativa.
- **Tendenza con la scala:**
  - Il distacco di C5 da A+calc resta fermo tra −3,4 e −4,0 punti.
  - Quello di A1 da A+calc resta tra +1,5 e −1,3 punti, ed è −0,2 sul 4B.
  - Il costo del canonico non si riduce con la scala; quello della prosa concisa resta nel rumore.
- **L'ipotesi regge per A1 e solo in parte per A2.**
  - A1 conserva il 70% del recupero dopo un'uguaglianza sbagliata (A 0,43 → A1 0,30 sull'1,7B, contro 0,09 di C5) e tutta l'accuratezza sui problemi con incognite.
  - A2, comprimendo di più (−36% di token), torna vicino a C5 (+2,3 punti sullo 0,6B, pari sull’1,7B): ne eredita il difetto, cioè righe senza etichetta che "perdono" il significato del numero (12 errori su 20 dovuti al formato, contro 0 su 20 di A1).
- **Il limite non sono le parole ma le cifre.** Con Qwen3 ogni cifra è un token: equazioni e risposta valgono circa metà di A. Togliere le parole di riempimento è gratis; togliere le parole di relazione no.
