## Fase J: 0,6B e 1,7B (commento)

**Setup.** Gli adapter A1 e A2 sono addestrati come in v5: LoRA r=16, α=32, lr 4e-4, 2 epoche, batch 32, 398 step. Sono 3 seed sullo 0,6B e 2 sull'1,7B, un processo per adapter. Il training dura circa 6,5 min per adapter sullo 0,6B e 9,5 min sull'1,7B. Le valutazioni coprono i 1319 problemi del test GSM8K in decodifica greedy. Le condizioni 1, 2, 3 e 5 sono riprese da v6 (run, output e latenze).

**Accuratezza e token** (media sui seed; tra parentesi l'IC 95% bootstrap):

| | A | A+calc | C5 | C5+gram+calc | A1 | **A1+calc** | A2 | A2+calc |
|---|---|---|---|---|---|---|---|---|
| 0,6B acc | 0,567 | 0,576 (0,553–0,600) | 0,524 | 0,537 | 0,577 | **0,592** (0,568–0,616) | 0,548 | 0,560 |
| 0,6B token | 99,9 | 101,1 | 76,1 | 75,9 | 75,9 | **75,1** | 64,6 | 64,4 |
| 1,7B acc | 0,724 | 0,725 (0,702–0,747) | 0,691 | 0,691 | 0,705 | **0,712** (0,688–0,734) | 0,693 | 0,691 |
| 1,7B token | 101,0 | 101,6 | 74,3 | 74,1 | 75,2 | **76,5** | 65,0 | 65,3 |

**Confronti McNemar (aggregati sui seed, con p per seed):**

| Modello | Confronto | Differenza | p | p per seed | Lettura |
|---|---|---|---|---|---|
| 0,6B | A1+calc vs A+calc | +1,5 punti | 0,044 | 0,35 / 0,019 / 0,95 | debole e non stabile tra i seed |
| 0,6B | A1+calc vs C5+gram+calc | +5,5 punti | 7·10⁻¹² | tutti < 0,001 | |
| 0,6B | A2+calc vs C5+gram+calc | +2,3 punti | 0,003 | | |
| 0,6B | A2+calc vs A+calc | −1,7 punti | 0,032 | | |
| 1,7B | A1+calc vs A+calc | −1,3 punti | 0,12 | 0,21 / 0,39 | non significativo |
| 1,7B | A1+calc vs C5+gram+calc | +2,1 punti | 0,028 | 0,06 / 0,24 | |
| 1,7B | A2+calc vs A+calc | −3,4 punti | 0,0002 | | |
| 1,7B | A2+calc vs C5+gram+calc | 0,0 punti | 1,0 | | stesso risultato con il 12% di token in meno |

Senza calcolatrice, A1 contro A vale +1,0 punti sullo 0,6B (p 0,20) e −1,9 sull'1,7B (p 0,027).

**Frontiera accuratezza/token:**
- **0,6B:** sulla frontiera restano solo A1+calc e A2+calc. A1+calc domina sia A+calc (più accurato con il 26% di token in meno) sia C5+gram+calc (+5,5 punti a parità di token).
- **1,7B:** la frontiera è A2 → A1 → A1+calc → A / A+calc. C5 e C5+gram+calc ne sono fuori: A1 senza calcolatrice li domina (+1,4 punti a parità di token), e A2 li raggiunge con 9 token in meno. A2+calc è fuori frontiera perché dominato da A2 senza calcolatrice.

**La calcolatrice:**
- **0,6B:** con A1 e A2 vale +1,5 e +1,2 punti, significativi su tutti i seed (p < 10⁻⁵).
- **1,7B:** vale +0,7 punti con A1 (p 0,04) e nulla con A2. Come in v6, il modello grande sbaglia pochi conti: senza calcolatrice solo il 4–6% delle risposte contiene un’uguaglianza sbagliata.
- Il numero medio di token forzati per risposta resta circa 11 in tutti i formati: la calcolatrice interviene sulle stesse equazioni, solo con meno parole intorno.

**Recupero dopo un'uguaglianza sbagliata** (condizioni senza calcolatrice; quota di problemi con almeno un `=` sbagliato che arrivano comunque alla risposta giusta):

| | A | A1 | A2 | C5 |
|---|---|---|---|---|
| 0,6B | 0,24 | 0,17 | 0,15 | 0,07 |
| 1,7B | 0,43 | 0,30 | 0,34 | 0,09 |

La prosa concisa conserva gran parte del recupero di A, il 60–80%. C5 ne perde quasi tutto. Le parole contano: dove resta una frase intorno al conto, il modello può "rileggere" e correggere la strada.

**Problemi con incognite (61):**
- **0,6B:** A1 vale 0,268, quanto A (0,257) e sopra C5 (0,186).
- **1,7B:** A1 vale 0,426, quanto A (0,434) e sopra C5 (0,361).
- A2 sta in mezzo: 0,20 e 0,37.

**Latenza** a batch 1 su 200 problemi, seed 0:
- 0,6B: A1+calc risponde in 1,23 s, A+calc in 1,60 s (−23%). A2+calc in 1,01 s (−37%).
- 1,7B: 1,46 s contro 1,70 s (−14%).

Il guadagno sull'1,7B è minore del calo dei token perché i ms per token delle condizioni nuove (18,2–18,6) superano quelli ripresi da v6 (16,6–16,7). È lo stesso modello e la stessa macchina, ma una sessione diversa: un ~10% di deriva di sistema, non un effetto del formato (vedi Limiti).

Anche il throughput a batch 32 dell'1,7B con calcolatrice (A1+calc 2,99 risposte/s) risente della stessa sessione. Per confrontare i formati, il conteggio dei token è la misura più robusta.

**Analisi degli errori (1,7B, seed 0, 20 errori per condizione)**

Ho usato lo stesso metodo di v6: un classificatore LLM separato (Sonnet), le stesse categorie, una categoria primaria per errore (il primo passo che diverge) e il flag "ambiguità del formato". Il file è `results/v7/errors_classified.csv`.

| | piano sbagliato | quantità confusa | passaggio mancante | aritmetico | formato rotto | altro | ambiguità del formato |
|---|---|---|---|---|---|---|---|
| 1,7B, 10 A1+calc | 9 | 6 | 2 | 1 | 0 | 2 | **0** su 20 |
| 1,7B, 12 A2+calc | 5 | 9 | 4 | 2 | 0 | 0 | **12** su 20 |
| v6: 1,7B, 2 A+calc | 8 | 6 | 2 | 0 | 0 | 4 | 0 su 20 |
| v6: 1,7B, 5 C5+gram+calc | 9 | 6 | 2 | 0 | 1 | 2 | 13 su 40 (0,6B + 1,7B) |

**Osservazioni:**
- **A1 sbaglia come A.** Gli errori sono soprattutto di piano, cioè di lettura del problema, e nessuno è dovuto al formato. Esempi:
  - test-1183: "so 2 + 5 = 7 apples fell" ignora la mela messa in tasca;
  - test-606: "725 * 2 = 1450 times per mile" moltiplica invece di dividere.
  La compressione di A1 toglie parole che non servivano al ragionamento.
- **A2 perde il significato dei numeri.** In 12 errori su 20 la riga senza etichetta, o senza la parola di relazione, è la causa plausibile: è il meccanismo di C5 (nomi che "mentono") in forma di prosa. Esempi:
  - test-749: `12 - 6 = 6` sottrae i 6 gruppi invece dei 9 formati;
  - test-203: `Sandwiches: 7 - 2 = 5` perde il "each" delle 2 fette;
  - test-495: `5 days * 5 = 25 minutes` perde il "per hour".
- **Solo in A2:** il modello non sa più quale riga è il risultato. In test-933 continua oltre il risultato giusto (`30 - 15 = 15`, poi `15 / 2 = 7.5`), in test-514 risponde con un valore intermedio (`x = 30000`).
- **In tutti e due i formati:** le percentuali scritte come decimali seguiti da "%" non vengono convertite (`0.6 * 100% = 0.6%`, test-1235). È un errore che la calcolatrice "certifica".
- **Nessun formato rotto,** né troncamenti né loop, contro i loop di C6 in v6.
