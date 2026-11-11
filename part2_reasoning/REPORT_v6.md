# Esperimento v6: canonico migliorato, decodifica vincolata e calcolatrice

<!-- FINALE -->

Codice in `v5/` (orchestratore `v5/run_v6.py`; moduli nuovi `src/spec6.py`, `src/grammar6.py`,
`src/constrained.py`, `src/eval6.py`, `src/analysis6.py`; 129 test). Risultati in `results/v6/`.
Modelli e adapter A/C5 sono quelli di v5 (`REPORT_v5.md`).

## Risposta in breve

- **No: il canonico migliorato non raggiunge A + calcolatrice.**
  - Nel confronto equo (condizione 8 contro 2, entrambe con calcolatrice) C6 perde **5,5 punti** sullo 0,6B (52,1% contro 57,6%) e **5,8** sull'1,7B (66,7% contro 72,5%), p < 0,001 in ogni seed.
  - Usa il **20–21% di token in meno** ed è più veloce: latenza a batch 1 0,75× sullo 0,6B e 0,84× sull'1,7B.
  - Il distacco è **più ampio** di quello di v5 tra C5 e A (4,3 e 3,3 punti).
  - Nella fase opzionale con **Qwen3-4B** (QLoRA, 1 seed) il distacco scende a 3,6 punti (79,2% contro 82,9%), con il 21% di token in meno.
- **Scomposizione del divario A − C5** (4,3 punti sullo 0,6B, 3,3 sull'1,7B):

  | Passaggio | 0,6B | 1,7B |
  |---|---|---|
  | Grammatica vincolata (3→4) | −0,9 | −1,1 |
  | Calcolatrice (4→5) | +2,2 | +1,0 |
  | Grammatica C6 (5→8) | −1,5 | −2,4 |

  Il miglior canonico è **C5 con grammatica e calcolatrice (condizione 5)**: 53,7% sullo 0,6B e 69,1% sull'1,7B. Resta comunque 3,9 e 3,4 punti sotto A + calcolatrice, come in v5.
- **Le tre cause di v5:**
  - **Output fuori grammatica:** eliminati dalla decodifica vincolata (conformità 98–99,5%; il resto sono loop troncati). Ma gli output "fuori grammatica" erano spesso **giusti**: tutti i problemi persi passando da 3 a 4 erano non conformi e corretti senza vincolo.
  - **Mancato recupero dagli errori di conto:** risolto, perché con la calcolatrice le uguaglianze sbagliate scendono a 0–0,2% in C. Vale +1–2 punti, ma aiuta anche A (+1,0 sullo 0,6B, +0,1 sull'1,7B), quindi non chiude il divario.
  - **Ambiguità dei nomi:** C6 la dimezza negli errori campionati (6 su 40 contro 13 su 40). Però costa il 21% dei dati di training, il 6–9% di token in più e porta nuovi errori (unità sbagliate, suffissi casuali). Nel complesso C6 è peggiore di C5.

## 1. La grammatica C6

C6 parte da C5 (sezione 2 di REPORT_v5) con queste modifiche:

```
calc    := NAME " = " EXPR " = " NUM
given   := "GIVEN " NAME " = " NUM
fact    := "FACT " NAME " = " NUM
check   := "IF " EXPR CMP EXPR
solve   := "SOLVE " NAME ": " EQ " = " EQ " -> " NAME " = " NUM     equazione lineare in UNA incognita
note    := "NOTE " parola{1,12}                                      al massimo 2 per soluzione
final   := "ANSWER = " NUM
NAME    := [a-z][a-z0-9]*(_[a-z0-9]+){0,4}_UNIT                      l'ultimo segmento è l'unità
UNIT    := usd pct n ratio s min h days weeks months years mi km m ft in lb oz kg l gal   (21)
EXPR    := come C5: solo numeri, + - * / e parentesi
EQ      := numeri, operatori e l'incognita
```

**Regole:**
- Ogni nome è definito una sola volta.
- Le tariffe mettono il denominatore nel nome e l'unità del valore in fondo (`pay_per_h_usd`).
- `_n` indica i conteggi puri, `_ratio` i fattori.
- Le percentuali sono nomi `_pct` con valore tra 0 e 100, usati come `x * p / 100`.

**Parser (`grammar6.check_c6`)** verifica:
- la grammatica e le unità della lista;
- che non ci siano ridefinizioni;
- `SOLVE` con una sola incognita, equazione lineare (controllata calcolando f in 0, 1 e 2) e valore uguale alla radice;
- il valore dei `_pct` tra 0 e 100;
- al massimo 2 `NOTE` di al massimo 12 parole;
- l'aritmetica e le condizioni `IF`, come in C5.

**Dati C6:**
- Generati con lo stesso generatore di v5 (Qwen3-4B-Instruct Q8 con llama.cpp), sui 6681 problemi di v5 (6381 di training e 300 di dev).
- Prompt con 4 esempi: uno con `FACT`, uno con percentuali, uno con `SOLVE` e `NOTE`, uno semplice.
- Normalizzazione e verifica come in v5. Generazione in 34 minuti.

| | A | C5 | C6 |
|---|---|---|---|
| Validi sui 6381 problemi di training | 6381 | 6381 (per costruzione) | **5065 (79%)** |
| Token medi per soluzione, sugli stessi 5065 | 92,5 | 69,2 | 74,4 (+7,5% rispetto a C5) |
| Uso di `NOTE` / `SOLVE` nei dati validi | – | – | 0% / 0,5% (24 soluzioni) |

**Scarti:** 1281 su 1316 sono di grammatica, soprattutto per l'unità non in fondo al nome (`profit_ratio_mike`) o fuori lista (`_ml`, `miles`).

**`NOTE` e `SOLVE` nella generazione:**
- Il generatore ha tentato `NOTE` solo 2 volte su 9825 generazioni, entrambe non valide (`NOTE 85 > 84`).
- Ha tentato `SOLVE` 118 volte, con 25 validi.
- I problemi con incognite erano già quasi tutti esclusi dal sottoinsieme v5, quindi quasi nessuno ne aveva bisogno.

**Unità nei dati:** 55% `_n`, 29% `_usd`, poi `_min`, `_h`, `_ft`, `_lb`… Negli output dell'1,7B: 56% `_n`, 28% `_usd`.

Esempi A / C5 / C6 dello stesso problema sono in `results/v6/examples_A_C5_C6.md`. Il primo esempio (train-5) è anche uno dei few-shot.

**Training C6:**
- Identico a C5: LoRA r = 16, α = 32, stessi moduli, lr 4e-4, 2 epoche, batch 32, loss solo sulla soluzione.
- Seed: 3 sullo 0,6B e 2 sull'1,7B.
- Con 5065 esempi le 2 epoche sono **318 step** invece di 400.

## 2. Decodifica vincolata e calcolatrice

**Implementazione:** fatta a mano come `LogitsProcessor` (`src/constrained.py`), una macchina a stati per riga del batch. Non ho provato librerie di decodifica a grammatica, perché la grammatica è a righe e regolare.

**Grammatica (C5 e C6):**
- La decodifica è greedy: a ogni passo il processore prende i candidati in ordine di logit (i primi 64, poi i primi 4096 se nessuno è valido) e tiene il **primo valido**. Con greedy questo equivale a mascherare tutti i token non validi.
- Un token è valido se:
  - ogni riga completata corrisponde per intero alla regex di riga, la riga aperta è un prefisso valido (match parziale del modulo `regex`) e il testo è solo ASCII;
  - non ridefinisce un nome già usato;
  - per C6, usa solo unità in lista e non apre una terza `NOTE`.
- EOS è ammesso solo dopo una riga `ANSWER = N` completa.

**Calcolatrice per C5 e C6:**
- Quando la riga è `NAME = EXPR =`, calcola EXPR con `ast` e forza i token di `" " + valore`. Qwen tokenizza lo spazio e le cifre una per una.
- Per `SOLVE … -> NAME =` risolve l'equazione e forza la radice.
- Dopo il valore è ammesso solo `\n`.
- Formato del valore: intero se lo è, altrimenti 2 decimali senza zeri finali.

**Calcolatrice per A:**
- Scatta quando il testo termina con un'espressione aritmetica seguita da `=`. Regex `A_CALC_TRIGGER` in `spec6.py`:
  ```
  (?<![\w.])(\(?\$?\d[\d,]*(\.\d+)?%?\)?(\s*[-+*/×x]\s*\(?\$?\d[\d,]*(\.\d+)?%?\)?)+)\s*=$
  ```
- Non scatta se l'espressione è la coda di un'espressione più lunga (testo precedente che finisce con un operatore).
- `%` diventa `/100`, tranne nelle somme e differenze di sole percentuali (`100% - 40% = 60`).
- Dopo il valore forzato il token successivo non può iniziare con una cifra.

**Verifica della calcolatrice di A su 200 soluzioni di training** (`a_calc_audit.json`):
- Sui 733 segni `=` la regex scatta in 671 casi (92%), e in **669 il valore coincide** con quello scritto.
- I 2 disaccordi sono una catena `(2 * 111) - 50 = 222 - 50` (valore forzato 172, comunque corretto) e il numero misto `8 x 1 1/2`.
- Non scatta quando tra i numeri ci sono parole o unità (`5 snakes/jaguar * 6 jaguars =`) o il segno `−` Unicode.
- Una prima versione forzava valori sbagliati su `students * 2/3 =` e su `100% - 40% =`; li ho corretti prima delle valutazioni.

**Test unitari** (`tests/test_constrained.py`, `tests/test_grammar6.py`):
- rifiuto di prosa, ridefinizioni, unità fuori lista, terza `NOTE`, EOS prima di `ANSWER` e newline dopo `ANSWER`;
- forzatura del valore in C5 e in `SOLVE`;
- la calcolatrice di A scatta e non scatta dove deve;
- batch con stato indipendente per riga.

## 3. Risultati: 8 condizioni × 2 modelli (GSM8K test, 1319 problemi, greedy)

**Condizioni di valutazione:**
- Accuratezza: media dei seed (3 sullo 0,6B, 2 sull'1,7B), con IC 95% bootstrap sui problemi.
- Latenza: batch 1, 200 problemi fissi, seed 0.
- Throughput: batch 32.
- Le condizioni 1 e 3 sono i risultati di v5.

| Modello | # | Condizione | Accuratezza [IC 95%] | Conformità | Token medi (mediana) | Forzati | Con uguaglianza sbagliata | Recupero | Latenza b=1 (s) | ms/token | Risposte/s b=32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0,6B | 1 | A | 56,7 [54,2–59,1] | 99,5% | 99,9 (91) | 0 | 9,0% | 24% | 1,502 | 14,8 | 4,01 |
| 0,6B | 2 | A + calc | **57,6** [55,3–60,0] | 99,0% | 101,1 (91) | 11,5 | 1,1% | 29% | 1,597 | 15,3 | 3,43 |
| 0,6B | 3 | C5 | 52,4 [50,0–54,8] | 91,9% | 76,1 (68) | 0 | 7,0% | 6% | 1,160 | 14,9 | 5,12 |
| 0,6B | 4 | C5 + gram | 51,4 [49,0–53,9] | 98,8% | 76,6 (68) | 0 | 6,9% | 5% | 1,216 | 15,3 | 4,54 |
| 0,6B | 5 | C5 + gram + calc | **53,7** [51,3–56,1] | 98,7% | 75,9 (68) | 11,3 | 0,1% | – | 1,206 | 15,4 | 4,63 |
| 0,6B | 6 | C6 | 51,7 [49,2–54,0] | 87,7% | 81,1 (73) | 0 | 7,3% | 5% | 1,188 | 14,7 | 5,17 |
| 0,6B | 7 | C6 + gram | 50,3 [47,9–52,7] | 98,4% | 80,4 (73) | 0 | 7,0% | 4% | 1,249 | 15,2 | 5,06 |
| 0,6B | 8 | C6 + gram + calc | 52,1 [49,8–54,5] | 98,5% | 79,7 (73) | 11,1 | 0,0% | – | 1,191 | 15,0 | 5,20 |
| 1,7B | 1 | A | 72,4 [70,2–74,6] | 99,8% | 101,0 (93) | 0 | 7,4% | 43% | 1,673 | 16,6 | 3,73 |
| 1,7B | 2 | A + calc | **72,5** [70,2–74,7] | 99,6% | 101,6 (93) | 11,3 | 1,1% | 45% | 1,699 | 16,7 | 3,33 |
| 1,7B | 3 | C5 | 69,1 [66,9–71,5] | 93,5% | 74,3 (68) | 0 | 3,5% | 9% | 1,141 | 15,5 | 5,05 |
| 1,7B | 4 | C5 + gram | 68,1 [65,8–70,4] | 99,5% | 74,8 (68) | 0 | 3,6% | 7% | 1,243 | 16,9 | 4,42 |
| 1,7B | 5 | C5 + gram + calc | **69,1** [66,9–71,5] | 99,5% | 74,1 (67) | 11,1 | 0,2% | – | 1,202 | 16,4 | 4,61 |
| 1,7B | 6 | C6 | 66,4 [64,1–68,7] | 89,4% | 80,9 (73) | 0 | 4,4% | 10% | 1,309 | 15,9 | 4,64 |
| 1,7B | 7 | C6 + gram | 65,1 [62,9–67,4] | 98,1% | 82,5 (74) | 0 | 4,7% | 7% | 1,380 | 16,1 | 3,77 |
| 1,7B | 8 | C6 + gram + calc | 66,7 [64,5–69,0] | 98,1% | 81,1 (73) | 11,1 | 0,1% | – | 1,425 | 16,8 | 3,99 |

**Come leggere la tabella:**
- "Con uguaglianza sbagliata" è la quota di output con almeno un `a op b = c` errato, misurata con la stessa regex di v5.
- "Recupero" è l'accuratezza tra quegli output. Con la calcolatrice in C restano 0–3 casi in tutto, quindi non è stimabile.
- In A + calc resta l'1,1% di uguaglianze sbagliate, nei punti dove la regex non scatta (parole o unità tra i numeri). La regex di misura è più permissiva di quella che fa scattare la calcolatrice.

**Uso di `NOTE` e `SOLVE` negli output C6:** **0%** in tutte le condizioni e in tutti i seed. Il modello non le ha imparate: `NOTE` è assente dai dati e `SOLVE` compare in 24 esempi.

**Problemi con incognite** (61 problemi del test con soluzione ufficiale algebrica, regex `ALGEBRA_RE`; sul train l'83% di questi falliva la riscrittura C5, contro il 7% degli altri):

| | A | A + calc | C5 | C5 + gram + calc | C6 | C6 + gram + calc |
|---|---|---|---|---|---|---|
| 0,6B | 25,7% | 25,1% | 18,6% | 14,2% | 20,2% | 16,4% |
| 1,7B | 43,4% | 41,8% | 36,1% | 34,4% | 33,6% | 31,1% |

C6 non recupera questi problemi, perché `SOLVE` non viene mai usato.

**Costo dei processori:**
- A batch 1, da +0,1 a +1,4 ms/token (+1–9%): A 14,8 → 15,3 e 16,6 → 16,7; C5 14,9 → 15,4 e 15,5 → 16,4–16,9.
- A batch 32 il throughput cala del 10–15% (A 4,01 → 3,43 e 3,73 → 3,33 risposte/s), perché il processore lavora in Python per riga e per passo.

**Grafici:** `results/v6/plots/acc_vs_tokens_{small,large}.png`, con accuratezza contro token e contro latenza, una per modello, 8 punti.

## 4. Scomposizione del divario e confronti appaiati

McNemar esatto, con seed accoppiati: p aggregato, poi p per seed.

| Modello | Passaggio | Δ accuratezza | p aggregato | p per seed |
|---|---|---|---|---|
| 0,6B | 3 → 4 grammatica | −0,9 | < 0,001 | 0,0004 / 0,035 / 0,002 |
| 0,6B | 4 → 5 calcolatrice | +2,2 | < 0,001 | < 0,001 in ogni seed |
| 0,6B | 5 → 8 C6 | −1,5 | 0,047 | 0,53 / 0,36 / 0,067 |
| 0,6B | 3 → 6 C6 senza decodifica | −0,7 | 0,36 | 0,56 / 0,82 / 0,52 |
| 0,6B | 1 → 2 calcolatrice su A | +1,0 | 0,001 | 0,32 / 0,029 / 0,032 |
| 0,6B | **2 → 8 (principale)** | **−5,5** | < 0,001 | 0,001 / 0,006 / < 0,001 |
| 1,7B | 3 → 4 grammatica | −1,1 | < 0,001 | 0,004 / 0,004 |
| 1,7B | 4 → 5 calcolatrice | +1,0 | < 0,001 | 0,001 / 0,002 |
| 1,7B | 5 → 8 C6 | −2,4 | 0,006 | 0,43 / 0,002 |
| 1,7B | 3 → 6 C6 senza decodifica | −2,8 | 0,002 | 0,51 / < 0,001 |
| 1,7B | 1 → 2 calcolatrice su A | +0,1 | 0,91 | 1 / 1 |
| 1,7B | **2 → 8 (principale)** | **−5,8** | < 0,001 | < 0,001 / < 0,001 |

**Scomposizione:**

| | Divario A − C5 (1 − 3) | Grammatica (3 → 4) | Calcolatrice (4 → 5) | Grammatica C6 (5 → 8) | Residuo 8 − 2 | Token 8/2 | Latenza 8/2 |
|---|---|---|---|---|---|---|---|
| 0,6B | 4,3 | −0,9 | +2,2 | −1,5 | −5,5 | 0,79 | 0,75 |
| 1,7B | 3,3 | −1,1 | +1,0 | −2,4 | −5,8 | 0,80 | 0,84 |

**Perché la grammatica vincolata costa:**
- Tutti i problemi persi tra 3 e 4 (43 sullo 0,6B e 36 sull'1,7B, seed sommati) erano, senza vincolo, **non conformi ma corretti**.
- Senza vincolo il 25% (0,6B) e il 50% (1,7B) degli output fuori grammatica erano comunque giusti: il modello "esce" dalla grammatica (ridefinisce un nome, usa un nome in un'espressione) proprio quando gli serve.
- Vincolato, è costretto a una strada che non sa percorrere.
- Solo 4 perdite su 79 sono loop troncati.

## 5. Analisi degli errori

**Metodo:**
- 20 errori campionati per le condizioni 2, 5 e 8 (seed 0), 120 in tutto.
- Stesso classificatore LLM separato di v5 (Sonnet) e stesse categorie, più due nuove.
- File: `results/v6/errors_classified.csv`.

| | aritmetico | passaggio mancante | quantità confusa | formato rotto | piano sbagliato | nome con unità sbagliata | NOTE fuorviante | altro | ambiguità del formato |
|---|---|---|---|---|---|---|---|---|---|
| 0,6B, 2 A + calc | 1 | 4 | 6 | 0 | 9 | – | – | 0 | 0 |
| 0,6B, 5 C5 + gram + calc | 1 | 6 | 8 | 1 | 4 | – | – | 0 | |
| 0,6B, 8 C6 + gram + calc | 0 | 2 | 5 | 0 | 12 | 1 | 0 | 0 | |
| 1,7B, 2 A + calc | 0 | 2 | 6 | 0 | 8 | – | – | 4 | 0 |
| 1,7B, 5 C5 + gram + calc | 0 | 2 | 6 | 1 | 9 | – | – | 2 | |
| 1,7B, 8 C6 + gram + calc | 1 | 3 | 4 | 1 | 10 | 1 | 0 | 0 | |

Errori da ambiguità del formato: **0** su 40 in A + calc, **13** su 40 in C5 + gram + calc, **6** su 40 in C6 + gram + calc.

**Osservazioni:**
- **Gli errori di piano diventano la prima causa** (circa metà). La calcolatrice ha tolto gli errori di conto, ma "certifica" anche operazioni sbagliate: test-1033, `15 - 54 / 18 = 12`, ha un errore di precedenza.
- **C6 dimezza l'ambiguità dei nomi** rispetto a C5. Nei casi C5 il nome "mente": `pizzas_per_person = 12` sono fette, una percentuale è trattata come dollari, un valore è diviso due volte per 100.
- **C6 introduce nuovi errori.**
  - Unità sbagliata: test-1259, `days_in_june_n = 30 * 24` sono ore, e la risposta è 172800 invece di 7200.
  - Nomi troncati che finiscono con un suffisso casuale: test-119, `…_4_years_l`.
  - Suffissi che nascondono la distinzione: `_usd` mensile contro una tantum; `_n` maniche contro scatole.
- **Loop indotti dal divieto di ridefinire:** il modello inventa nomi sempre più lunghi (`…_after2doubled2doubled…`) fino al limite di token (test-1216, test-389).
- **`NOTE` e `SOLVE` non compaiono mai** negli errori della condizione 8. Almeno 5 errori sono problemi con un'incognita, risolti con una scorciatoia sbagliata invece che con `SOLVE`.
- 6 casi sono "altro" per riferimenti ambigui o sbagliati (per esempio test-403: la risposta del modello è corretta).

## 6. Fase opzionale: Qwen3-4B-Base con QLoRA

**Setup del 4B:**
- Qwen3-4B-Base in 4 bit NF4 (bitsandbytes 0.50.2, funziona su Windows), LoRA con gli stessi iperparametri.
- Gradient checkpointing, micro-batch 4: senza checkpointing la VRAM saturava e il training riversava in RAM.
- 1 seed; adapter A (400 step, 40 minuti) e C6 (318 step, 26 minuti).
- Valutazione con l'adapter non fuso sui pesi a 4 bit.
- Condizioni 2 e 8.

| | Accuratezza [IC 95%] | Conformità | Token | Con incognite | Latenza b=1 | ms/token | Risposte/s b=32 |
|---|---|---|---|---|---|---|---|
| 2 A + calc | **82,9** [80,9–84,8] | 99,2% | 102,9 | 59,0% | 5,08 s | 48,5 | 1,14 |
| 8 C6 + gram + calc | 79,2 [77,0–81,5] | 99,2% | 81,4 | 44,3% | 3,95 s | 48,5 | 1,52 |

Differenza −3,6 punti (McNemar p = 0,001), con il 21% di token in meno e una latenza 0,78× (1,29× più veloce).

**Tendenza con la scala del distacco 8 − 2:** −5,5 (0,6B), −5,8 (1,7B), **−3,6 (4B)**. Il distacco si riduce passando al 4B ma non tra 0,6B e 1,7B. In v5 il distacco tra C5 e A scendeva già da 4,3 a 3,3.

Va letto con cautela:
- c'è un solo seed;
- il 4B è quantizzato a 4 bit (regime QLoRA diverso);
- la latenza assoluta è dominata dalla dequantizzazione (48 ms/token), ma il rapporto tra formati segue ancora i token.

Sui problemi con incognite il distacco resta grande (−15 punti): `SOLVE` non viene usato nemmeno dal 4B.

## 7. Risposta alla domanda principale (condizione 8 contro 2)

**Il canonico migliorato raggiunge A + calcolatrice con meno token? No.**

| Modello | Accuratezza (8 contro 2) | Token | Latenza a batch 1 | Guadagno del ragionamento conservato |
|---|---|---|---|---|
| 0,6B | 52,1% contro 57,6% | −21% | 0,75× | 87% |
| 1,7B | 66,7% contro 72,5% | −20% | 0,84× | 88% |

Il guadagno conservato è calcolato rispetto a D (v5).

**Frontiera accuratezza / token / latenza** (grafici in `plots/`):
- Su entrambi i modelli la frontiera è formata da **A + calc** (massima accuratezza) e **C5 + grammatica + calcolatrice** (condizione 5).
- La condizione 5 tiene il 90–93% del guadagno del ragionamento con il 25–27% di token in meno ed è 1,32–1,41× più veloce a batch 1.
- C6 è dominato da C5 in tutte e tre le condizioni corrispondenti: meno accurato e con più token. È anche più lento, salvo la condizione 8 sullo 0,6B, che è pari alla 5 (1,19 contro 1,21 s).

## 8. Confronto con v5

| v5 | v6 |
|---|---|
| C perde 4,3 e 3,3 punti rispetto ad A, con il 24–26% di token in meno | Con decodifica vincolata e calcolatrice il miglior C (condizione 5) perde ancora **3,9 e 3,4 punti** rispetto ad A + calc: il divario non si chiude |
| Il 6,5–8% di output è fuori grammatica | Eliminato dalla grammatica vincolata, ma senza guadagno: quegli output erano spesso giusti, e vincolarli costa 1 punto |
| In C un conto sbagliato quasi non si recupera (6–9%) | La calcolatrice elimina i conti sbagliati in C (+1–2 punti), ma migliora anche A, sullo 0,6B |
| L'ambiguità si sposta nei nomi | Imporre l'unità dimezza l'ambiguità dei nomi, ma costa dati (−21% di esempi validi) e token (+7%) e porta errori di unità. Nel complesso C6 è peggiore di C5 di 1,5–2,4 punti |
| C non esprime le incognite | `SOLVE` è stato aggiunto ma non usato: senza esempi nei dati (24) il modello non lo impara |

**Conclusione comune:** il vantaggio pratico della forma canonica sta nei token e nella latenza (1,3–1,4×), non nell'accuratezza. I miglioramenti di decodifica aiutano poco e in parte anche la base naturale.

## 9. Limiti e deviazioni

- **Dati C6 ridotti:**
  - C6 è valido sul 79% dei problemi di v5, quindi è addestrato su 5065 esempi invece di 6381 e fa 318 step invece di 400 (2 epoche come richiesto).
  - Una parte del calo di C6 può venire dai dati in meno e dalla selezione dei problemi "facili da canonicalizzare", non solo dalla grammatica.
- **`NOTE` e `SOLVE` non messi alla prova:**
  - Il generatore non ha usato `NOTE` (4 esempi few-shot, uno solo con `NOTE`).
  - Il sottoinsieme di problemi imposto (quelli di v5) esclude quasi tutti i problemi con incognite, quindi `SOLVE` è in 24 esempi.
  - La misura sui problemi con incognite si basa su un'euristica: 61 problemi del test riconosciuti da una regex sulla soluzione ufficiale.
- **Decodifica vincolata:**
  - È esatta solo per il greedy: il token scelto è il migliore valido tra i primi 64, poi tra i primi 4096; se non ce n'è nessuno viene forzato EOS.
  - Il vincolo `_pct` tra 0 e 100 è verificato solo nei dati, non in decodifica.
  - `pay_per_min` è accettato perché `_min` è un'unità: il suffisso non può distinguere nome e unità.
- **Calcolatrice di A:** copre il 92% delle uguaglianze dei dati di training. Non scatta con unità o parole tra i numeri, e su 2 casi rari (catene, numeri misti) forza un valore diverso da quello scritto.
- **Costo misurato dei processori:** è quello di un'implementazione Python per riga, quindi il throughput a batch 32 cala del 10–15%. Una versione vettorizzata o compilata costerebbe meno.
- **Latenze delle condizioni 1 e 3:** vengono dalla sessione v5 (stesso codice di generazione, stessi pesi). Le altre sono misurate in v6.
- **Adapter e seed:** gli adapter A e C5 sono quelli di v5, quindi le condizioni 2, 4 e 5 non richiedono training. Seed 3 e 2, McNemar aggregato ottimista, p per seed riportati.
- **Problemi tecnici:**
  - Il training di più adapter nello stesso processo non liberava la VRAM: il terzo adapter finiva in RAM di sistema ed era molto lento.
  - L'ho rilanciato con un processo per adapter, perdendo ~40 minuti; i risultati non sono toccati.
  - La catena automatica della fase 4B non è partita da sola ed è stata rilanciata a mano.
  - Il primo training del 4B senza gradient checkpointing saturava la VRAM: l'ho fermato e rilanciato con checkpointing.
- **Fase 4B:** 1 seed, solo condizioni 2 e 8, QLoRA a 4 bit invece di LoRA su bf16; non è direttamente confrontabile con gli altri modelli.
- **Tempo totale di v6:** ~8 h. Comprende la generazione C6 (34 minuti), le valutazioni (~4,5 h), il training C6 (~25 minuti) e la fase 4B (~2,3 h).
- **Analisi degli errori:** 20 errori per cella, classificati da un LLM; conteggi indicativi.
- **Euristica `ALGEBRA_RE`:** può includere qualche falso positivo (per esempio "I = …").

## File

| Contenuto | Percorso |
|---|---|
| Dati e statistiche C6 | `v5/data/rewrites/C6.jsonl`, `v5/data/sft/C6_train.jsonl`, `results/v6/data_c6.json`, `results/v6/examples_A_C5_C6.md` |
| Verifica della calcolatrice di A | `results/v6/a_calc_audit.json` |
| Run e latenze | `results/v6/runs.csv`, `results/v6/latency.csv`, `results/v6/outputs/*` |
| Analisi | `results/v6/summary.csv`, `results/v6/comparisons.csv`, `results/v6/decomposition.csv` |
| Errori | `results/v6/errors_sample.jsonl`, `results/v6/errors_classified.csv` |
| Grafici | `results/v6/plots/acc_vs_tokens_{small,large}.png` |
