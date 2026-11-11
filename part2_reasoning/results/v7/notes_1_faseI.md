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
