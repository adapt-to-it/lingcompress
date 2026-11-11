# Esperimento v7: prosa concisa

<!-- FINALE -->

**Stato: completato** (Fasi I, J, K). Le tabelle in fondo (Riepilogo, Frontiera, Confronti, Recupero, Scala) sono generate da `results/v7/`.

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


## Fase I: dati A1 / A2

Riscritture delle soluzioni A di v5 con Qwen3-4B-Instruct (Q8, llama.cpp) su 6381 problemi train + 300 dev. Sono ammessi fino a 3 tentativi: il primo a T=0, i successivi a T=0,7. Quando un tentativo è bocciato solo per lunghezza, il successivo riceve una richiesta esplicita di accorciare. Tempo di generazione: A1 34 min, A2 43 min.

**Deviazione dalla specifica: selezione "morbida".** La soglia di lunghezza (A1 ≤ 85%, A2 ≤ 65% dei token di A) è strutturalmente difficile da rispettare. Qwen3 tokenizza ogni cifra separatamente, così le sole equazioni più la riga finale valgono in media il 49% dei token di A (p90 0,67). Per il 12% dei problemi il limite di A2 è quasi impossibile da rispettare senza cambiare i conti.

Applicando la regola alla lettera ("addestra comunque sui validi"), A2 resterebbe con il 47% dei dati, concentrati sui problemi con meno numeri. È un bias di selezione che renderebbe il confronto con A e C5 non equo.

Ho quindi usato questa regola: si prende il primo tentativo valido; se nessuno lo è, si prende il tentativo più corto bocciato solo per lunghezza. Risposta e passaggi ufficiali sono comunque verificati.

| | A1 | A2 |
|---|---|---|
| problemi train | 6381 | 6381 |
| validi in senso stretto (tutti i controlli) | 5569 (87,3%) | 2975 (46,6%) |
| aggiunti solo per lunghezza | 774 | 3378 |
| tenuti (training) | 6343 (99,4%) | 6353 (99,6%) |
| scartati (risposta / passaggi / formato) | 16 / 21 / 2 | 10 / 21 / 0 |
| rapporto medio token vs A, testi selezionati (tutto train+dev) | 0,755 | 0,676 |
| obiettivo della specifica | ≈ 0,75 | ≈ 0,55 |

La copertura stretta è sotto l'obiettivo del 95%: A1 87%, A2 47%. Gli id scartati sono in `results/v7/dropped_A1.csv` e `dropped_A2.csv`; l'elenco degli id validi in senso stretto si ricava da `data/rewrites/A{1,2}.jsonl`. A1 raggiunge l'obiettivo di compressione. A2 si ferma a −32% invece di −45%: il pavimento sono i numeri, non le parole.

**Token medi (Qwen3) sui 6329 problemi comuni ad A, A1, A2 e C5:**

| formato | token medi | parole per riga (escluse le equazioni) | quota di righe con parole di relazione |
|---|---|---|---|
| A | 95,1 | 9,5 | 0,56 |
| A1 | 70,7 (−26%) | 4,1 | 0,33 |
| A2 | 62,8 (−34%) | 1,6 | 0,21 |
| C5 | 70,9 (−25%) | 2,6 | 0,29 |

**A1 costa quanto C5 in token**, quindi a parità di accuratezza la frontiera la decide solo la precisione. A2 è il formato più corto dei quattro.

**Audit della calcolatrice A** su 200 soluzioni di training per formato: scatta sul 96,1% degli `=` in A1 (2 disaccordi su 706) e sul 96,7% in A2 (1 su 709). La soglia di 0,90 è superata.

**Esempio (train-5)**, gli altri due problemi sono in `results/v7/examples_A_A1_A2_C5.md`:

```
A : There are 80/100 * 10 = 8 more purple flowers than yellow flowers.
    So in Mark's garden, there are 10 + 8 = 18 purple flowers.
    Purple and yellow flowers sum up to 10 + 18 = 28 flowers.
    That means in Mark's garden there are 25/100 * 28 = 7 green flowers.
    So in total Mark has 28 + 7 = 35 plants in his garden.
    The answer is 35.
A1: There are 80/100 * 10 = 8 more purple flowers than yellow.
    So there are 10 + 8 = 18 purple flowers.
    Yellow and purple together total 10 + 18 = 28.
    Green flowers are 25/100 * 28 = 7.
    Total flowers are 28 + 7 = 35.
    The answer is 35.
A2: 80/100 * 10 = 8 more purple.
    10 + 8 = 18 purple.
    10 + 18 = 28 yellow and purple.
    25/100 * 28 = 7 green.
    28 + 7 = 35.
    The answer is 35.
C5: GIVEN yellow_flowers = 10
    purple_excess = 80 / 100 * 10 = 8
    purple_flowers = 10 + 8 = 18
    yellow_plus_purple = 10 + 18 = 28
    green_flowers = 25 / 100 * 28 = 7
    total_flowers = 28 + 7 = 35
    ANSWER = 35
```

A2 ha preso spesso la forma "conto, poi etichetta" (`10 + 8 = 18 purple.`) invece di "etichetta: conto" come negli esempi del prompt. L'etichetta dopo il risultato resta linguaggio naturale, ma il modello deve scegliere il conto prima di nominare la quantità.

**Stima dei tempi dopo la Fase I.**
- Fase J: 10 adapter, circa 1 h di training, e 20 valutazioni più 4 latenze, circa 2,5 h.
- Fase K: 2 adapter QLoRA da 4B, circa 1,5 h, e 2 valutazioni più latenze, circa 1 h.
- Totale circa 6 h, sotto le 10 h: si prosegue senza chiedere conferma.


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


## Riepilogo

| model | cond | label | n_seeds | acc | acc_lo | acc_hi | conform_rate | tok_mean | tok_median | forced_mean | frac_wrong_eq | recovery | acc_algebra | note_rate | solve_rate | lat_mean_s | ms_per_token_mean | answers_per_s | tokens_per_s | speedup_vs_c1 | frontier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small | 1 | A | 3 | 0.567 | 0.542 | 0.591 | 0.995 | 99.933 | 91.000 | 0.000 | 0.090 | 0.242 | 0.257 | 0.000 | 0.000 | 1.502 | 14.767 | 4.009 | 400.608 | 1.000 | False |
| small | 2 | A+calc | 3 | 0.576 | 0.553 | 0.600 | 0.990 | 101.091 | 91.000 | 11.542 | 0.011 | 0.289 | 0.251 | 0.000 | 0.000 | 1.597 | 15.289 | 3.425 | 346.117 | 0.941 | False |
| small | 3 | C5 | 3 | 0.524 | 0.500 | 0.548 | 0.919 | 76.133 | 68.000 | 0.000 | 0.070 | 0.065 | 0.186 | 0.000 | 0.000 | 1.160 | 14.940 | 5.121 | 389.523 | 1.295 | False |
| small | 5 | C5+gram+calc | 3 | 0.537 | 0.513 | 0.561 | 0.987 | 75.884 | 68.000 | 11.305 | 0.001 | 0.000 | 0.142 | 0.000 | 0.000 | 1.206 | 15.355 | 4.633 | 351.506 | 1.245 | False |
| small | 9 | A1 | 3 | 0.577 | 0.552 | 0.601 | 0.996 | 75.887 | 69.000 | 0.000 | 0.081 | 0.172 | 0.268 | 0.000 | 0.000 | 1.158 | 15.080 | 4.291 | 325.578 | 1.297 | False |
| small | 10 | A1+calc | 3 | 0.592 | 0.568 | 0.616 | 0.998 | 75.083 | 69.000 | 11.322 | 0.005 | 0.286 | 0.268 | 0.000 | 0.000 | 1.227 | 15.815 | 4.257 | 319.648 | 1.225 | True |
| small | 11 | A2 | 3 | 0.548 | 0.523 | 0.572 | 0.997 | 64.646 | 60.000 | 0.000 | 0.066 | 0.148 | 0.202 | 0.000 | 0.000 | 0.971 | 15.239 | 5.338 | 345.102 | 1.547 | False |
| small | 12 | A2+calc | 3 | 0.560 | 0.534 | 0.584 | 0.998 | 64.418 | 60.000 | 11.216 | 0.002 | 0.000 | 0.202 | 0.000 | 0.000 | 1.008 | 15.677 | 4.947 | 318.719 | 1.490 | True |
| large | 1 | A | 2 | 0.724 | 0.702 | 0.746 | 0.998 | 100.992 | 93.000 | 0.000 | 0.074 | 0.426 | 0.434 | 0.000 | 0.000 | 1.673 | 16.605 | 3.726 | 376.228 | 1.000 | True |
| large | 2 | A+calc | 2 | 0.725 | 0.702 | 0.747 | 0.996 | 101.560 | 93.000 | 11.315 | 0.011 | 0.448 | 0.418 | 0.000 | 0.000 | 1.699 | 16.682 | 3.327 | 337.728 | 0.985 | True |
| large | 3 | C5 | 2 | 0.691 | 0.669 | 0.715 | 0.935 | 74.264 | 68.000 | 0.000 | 0.035 | 0.087 | 0.361 | 0.000 | 0.000 | 1.141 | 15.540 | 5.045 | 374.566 | 1.466 | False |
| large | 5 | C5+gram+calc | 2 | 0.691 | 0.669 | 0.715 | 0.995 | 74.056 | 67.000 | 11.086 | 0.002 | 0.500 | 0.344 | 0.000 | 0.000 | 1.202 | 16.413 | 4.613 | 341.632 | 1.392 | False |
| large | 9 | A1 | 2 | 0.705 | 0.681 | 0.726 | 0.998 | 75.246 | 69.000 | 0.000 | 0.056 | 0.304 | 0.426 | 0.000 | 0.000 | 1.385 | 18.166 | 4.154 | 312.565 | 1.208 | True |
| large | 10 | A1+calc | 2 | 0.712 | 0.688 | 0.734 | 0.993 | 76.465 | 69.000 | 11.829 | 0.008 | 0.182 | 0.426 | 0.000 | 0.000 | 1.455 | 18.427 | 2.987 | 228.404 | 1.150 | True |
| large | 11 | A2 | 2 | 0.693 | 0.670 | 0.716 | 0.997 | 64.990 | 60.000 | 0.000 | 0.045 | 0.339 | 0.369 | 0.000 | 0.000 | 1.252 | 18.019 | 4.722 | 306.856 | 1.336 | True |
| large | 12 | A2+calc | 2 | 0.691 | 0.668 | 0.715 | 0.996 | 65.276 | 60.000 | 11.339 | 0.004 | 0.300 | 0.352 | 0.000 | 0.000 | 1.236 | 18.580 | 3.924 | 256.130 | 1.353 | False |
| xl | 2 | A+calc | 1 | 0.829 | 0.809 | 0.848 | 0.992 | 102.861 | 94.000 | 11.754 | 0.009 | 0.667 | 0.590 | 0.000 | 0.000 | 5.077 | 48.455 | 1.139 | 117.197 | - | True |
| xl | 5 | C5+gram+calc | 1 | 0.794 | 0.770 | 0.815 | 0.998 | 75.524 | 69.000 | 11.230 | 0.002 | 1.000 | 0.393 | 0.000 | 0.000 | 3.635 | 48.486 | 1.721 | 129.998 | - | True |
| xl | 8 | C6+gram+calc | 1 | 0.792 | 0.770 | 0.815 | 0.992 | 81.438 | 74.000 | 11.274 | 0.000 | - | 0.443 | 0.000 | 0.000 | 3.948 | 48.531 | 1.518 | 123.633 | - | False |
| xl | 10 | A1+calc | 1 | 0.826 | 0.806 | 0.848 | 0.995 | 75.817 | 69.000 | 11.558 | 0.003 | 0.500 | 0.574 | 0.000 | 0.000 | 3.730 | 47.983 | 1.456 | 110.416 | - | True |

## Frontiera

| model | cond | label | acc | tok_mean | lat_mean_s |
|---|---|---|---|---|---|
| small | 10 | A1+calc | 0.592 | 75.083 | 1.227 |
| small | 12 | A2+calc | 0.560 | 64.418 | 1.008 |
| large | 1 | A | 0.724 | 100.992 | 1.673 |
| large | 2 | A+calc | 0.725 | 101.560 | 1.699 |
| large | 9 | A1 | 0.705 | 75.246 | 1.385 |
| large | 10 | A1+calc | 0.712 | 76.465 | 1.455 |
| large | 11 | A2 | 0.693 | 64.990 | 1.252 |
| xl | 2 | A+calc | 0.829 | 102.861 | 5.077 |
| xl | 5 | C5+gram+calc | 0.794 | 75.524 | 3.635 |
| xl | 10 | A1+calc | 0.826 | 75.817 | 3.730 |

## Confronti a coppie

| model | cond_a | cond_b | label_a | label_b | n_cells | acc_diff | n01 | n10 | p | p_per_seed |
|---|---|---|---|---|---|---|---|---|---|---|
| small | 2 | 10 | A+calc | A1+calc | 3957 | 0.015 | 476 | 415 | 0.044 | 0.348 / 0.0189 / 0.954 |
| small | 2 | 12 | A+calc | A2+calc | 3957 | -0.017 | 443 | 510 | 0.032 | 0.457 / 0.284 / 0.0793 |
| small | 5 | 10 | C5+gram+calc | A1+calc | 3957 | 0.055 | 618 | 399 | 0.000 | 0.000338 / 4.67e-05 / 3.76e-05 |
| small | 5 | 12 | C5+gram+calc | A2+calc | 3957 | 0.023 | 515 | 424 | 0.003 | 0.0459 / 0.432 / 0.0276 |
| small | 10 | 12 | A1+calc | A2+calc | 3957 | -0.032 | 376 | 504 | 0.000 | 0.0713 / 0.000449 / 0.0491 |
| small | 9 | 10 | A1 | A1+calc | 3957 | 0.015 | 89 | 29 | 0.000 | 0.000313 / 0.00256 / 0.00661 |
| small | 11 | 12 | A2 | A2+calc | 3957 | 0.012 | 73 | 27 | 0.000 | 0.0614 / 0.0428 / 0.000431 |
| small | 1 | 9 | A | A1 | 3957 | 0.010 | 476 | 436 | 0.197 | 0.908 / 0.0321 / 0.954 |
| small | 1 | 11 | A | A2 | 3957 | -0.019 | 448 | 522 | 0.019 | 0.364 / 0.402 / 0.0297 |
| small | 2 | 5 | A+calc | C5+gram+calc | 3957 | -0.040 | 447 | 605 | 0.000 | 0.00872 / 0.0774 / 7.12e-05 |
| small | 3 | 9 | C5 | A1 | 3957 | 0.053 | 624 | 414 | 0.000 | 0.00264 / 1.82e-05 / 0.000117 |
| small | 3 | 11 | C5 | A2 | 3957 | 0.024 | 534 | 438 | 0.002 | 0.0481 / 0.15 / 0.0784 |
| large | 2 | 10 | A+calc | A1+calc | 2638 | -0.013 | 227 | 262 | 0.124 | 0.212 / 0.393 |
| large | 2 | 12 | A+calc | A2+calc | 2638 | -0.034 | 232 | 321 | 0.000 | 0.00143 / 0.0401 |
| large | 5 | 10 | C5+gram+calc | A1+calc | 2638 | 0.021 | 330 | 275 | 0.028 | 0.0587 / 0.239 |
| large | 5 | 12 | C5+gram+calc | A2+calc | 2638 | 0.000 | 285 | 284 | 1.000 | 1 / 1 |
| large | 10 | 12 | A1+calc | A2+calc | 2638 | -0.020 | 239 | 293 | 0.021 | 0.0449 / 0.229 |
| large | 9 | 10 | A1 | A1+calc | 2638 | 0.007 | 43 | 25 | 0.038 | 0.243 / 0.11 |
| large | 11 | 12 | A2 | A2+calc | 2638 | -0.002 | 30 | 35 | 0.620 | 1 / 0.585 |
| large | 1 | 9 | A | A1 | 2638 | -0.019 | 232 | 283 | 0.027 | 0.0964 / 0.165 |
| large | 1 | 11 | A | A2 | 2638 | -0.031 | 243 | 325 | 0.001 | 0.00242 / 0.0851 |
| large | 2 | 5 | A+calc | C5+gram+calc | 2638 | -0.034 | 255 | 345 | 0.000 | 0.00194 / 0.0452 |
| large | 3 | 9 | C5 | A1 | 2638 | 0.014 | 322 | 286 | 0.156 | 0.141 / 0.615 |
| large | 3 | 11 | C5 | A2 | 2638 | 0.002 | 285 | 280 | 0.866 | 0.953 / 0.905 |
| xl | 2 | 10 | A+calc | A1+calc | 1319 | -0.002 | 92 | 95 | 0.884 | 0.884 |
| xl | 5 | 10 | C5+gram+calc | A1+calc | 1319 | 0.033 | 139 | 96 | 0.006 | 0.00603 |
| xl | 2 | 5 | A+calc | C5+gram+calc | 1319 | -0.035 | 98 | 144 | 0.004 | 0.00373 |
| xl | 2 | 8 | A+calc | C6+gram+calc | 1319 | -0.036 | 80 | 128 | 0.001 | 0.00107 |
| xl | 8 | 10 | C6+gram+calc | A1+calc | 1319 | 0.034 | 126 | 81 | 0.002 | 0.00215 |

## Recupero

| model | cond | label | frac_wrong_eq | recovery |
|---|---|---|---|---|
| small | 1 | A | 0.090 | 0.242 |
| small | 3 | C5 | 0.070 | 0.065 |
| small | 9 | A1 | 0.081 | 0.172 |
| small | 11 | A2 | 0.066 | 0.148 |
| large | 1 | A | 0.074 | 0.426 |
| large | 3 | C5 | 0.035 | 0.087 |
| large | 9 | A1 | 0.056 | 0.304 |
| large | 11 | A2 | 0.045 | 0.339 |

## Scala

| model | acc_2 | acc_5 | gap_5_2 | best_concise | acc_best | gap_best_2 | tok_ratio_5_2 | tok_ratio_best_2 |
|---|---|---|---|---|---|---|---|---|
| small | 0.576 | 0.537 | -0.040 | 10 | 0.592 | 0.015 | 0.751 | 0.743 |
| large | 0.725 | 0.691 | -0.034 | 10 | 0.712 | -0.013 | 0.729 | 0.753 |
| xl | 0.829 | 0.794 | -0.035 | 10 | 0.826 | -0.002 | 0.734 | 0.737 |

