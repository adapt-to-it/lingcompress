# Esperimento v5: ragionamento compresso su linguaggio reale (GSM8K + LoRA)

<!-- FINALE -->

Codice in `v5/` (orchestratore `v5/run_v5.py`, moduli in `v5/src/`, 68 test). Dati e risultati in `results/v5/`.

## Risposta in breve

- **Il ragionamento compresso mantiene quasi tutta l'accuratezza, non tutta.**
  - Il ragionamento vale molto: rispetto al solo numero (D) il formato pieno A guadagna 40 punti sul modello da 0,6B e 49 su quello da 1,7B.
  - Il telegrafico B conserva il **95–97%** di quel guadagno con il **12–13% di token in meno**.
  - Il canonico C ne conserva l'**89–93%** con il **24–26% di token in meno**.
  - In valore assoluto B perde 1,4 punti (0,6B, non significativo) e 2,6 punti (1,7B, significativo); C perde 4,3 e 3,3 punti (entrambi significativi).
- **È più veloce in proporzione ai token**, perché il costo per token non cambia (~15–17 ms con batch 1).
  - Latenza a batch 1: B è **1,14–1,20×** più veloce, C **1,30–1,47×**.
  - Throughput a batch 32: B 1,20–1,25× risposte/s, C **1,28–1,35×**.
- **Canonico contro telegrafico: dipende dal modello.**
  - Sull'1,7B il canonico è **migliore**: stessa accuratezza (69,1% contro 69,9%, p = 0,47) con il 15% di token in meno ed è 1,22× più veloce.
  - Sullo 0,6B C perde 2,9 punti rispetto a B (p < 0,001 aggregato; per seed p = 0,12 / 0,006 / 0,08).
  - Il canonico costa conformità: l'8% degli output dello 0,6B e il 6,5% dell'1,7B viola la grammatica, e la regolarità non toglie l'ambiguità ma la sposta nei **nomi delle quantità**.
- **Rispetto a v4:** la compressione paga in token e velocità, come previsto. La regolarità invece non migliora l'accuratezza: su modelli pre-addestrati il linguaggio naturale resta il formato "nativo".

## 1. Setup

| Voce | Scelta |
|---|---|
| Modelli | **Qwen3-0.6B-Base** ("small") e **Qwen3-1.7B-Base** ("large"), Apache 2.0, transformer puro |
| Modelli scartati | Qwen3.5-0.8B/2B-Base (feb. 2026): più recenti ma ibridi (attenzione lineare Gated DeltaNet) e multimodali, senza supporto affidabile su Windows per la KV cache statica |
| Ambiente | RTX 4070 SUPER 12 GB, Windows, torch 2.6, transformers 5.22.2, peft 0.21.2; SDPA e bf16 |
| Generatore dei dati | Niente API (manca `ANTHROPIC_API_KEY`): **Qwen3-4B-Instruct-2507 Q8_0** in locale con il server **llama.cpp** b12211 (CUDA), 24 slot. Scelta approvata |
| Prompt | Identico per tutti i formati: `Question: {domanda}\nAnswer:\n`. La loss è calcolata solo sui token della soluzione più EOS |

Nota sul generatore:
- Con transformers il 4B era troppo lento: ~1,8 s per problema, perché pesi e KV cache non entravano nei 12 GB e Windows riversava nella RAM.
- Il pilota è in `results/v5/pilot_generazione.md`.

## 2. La lingua canonica (formato C)

Grammatica definita prima della generazione: una riga per passaggio, nessuna riga vuota.

```
calc    := NAME " = " EXPR " = " NUM          passaggio aritmetico
given   := "GIVEN " NAME " = " NUM            quantità del testo ripresa senza calcolo
fact    := "FACT " NAME " = " NUM             costante di conoscenza comune (60 minuti in un'ora, 7 giorni ...)
check   := "IF " EXPR CMP EXPR                condizione vera che sceglie il caso (CMP in > < >= <= ==)
final   := "ANSWER = " NUM                    ultima riga, una sola volta
NAME    := [a-z][a-z0-9]*(_[a-z0-9]+){0,4}    snake_case, al massimo 5 parole, definito una sola volta
EXPR    := numeri, + - * / e parentesi; nessun nome, nessuna unità
NUM     := -?[0-9]+(\.[0-9]+)?                niente virgole, $, %
```

**Verifica (`v5/src/formats.py`, `check_c`):**
- Il parser controlla la grammatica e che nessun nome sia ripetuto.
- Ogni espressione è ricalcolata con `ast` e deve dare il valore scritto. La tolleranza è 1e-6, oppure l'arrotondamento se il valore è scritto con almeno 2 decimali.
- Le condizioni `IF` devono risultare vere.

**Normalizzazione deterministica**, applicata prima della verifica:
- C: `nome = numero` diventa `GIVEN nome = numero`; `18.00` diventa `18`; `$` e virgole delle migliaia vengono tolti.
- A: la riga finale è resa esattamente `The answer is N.`
- Senza normalizzazione C passava nel 58% dei casi, con normalizzazione nel 91%.

**Altri formati:**
- **A** pieno: frasi complete con i conti `x op y = z` e riga finale `The answer is N.`
- **B** telegrafico: A passato nello stripper a lista.
  - Toglie articoli (a, an, the), ausiliari e modali (is, are, was, has, do, will, can, could…), pronomi soggetto e loro contrazioni (I, you, he, she, it, we, they) e preposizioni semplici (of, to, in, on, at, by, with, from, for).
  - Restano numeri, operatori e i connettivi so, then, because, each, per, left, more, less, times, total.
- **D** solo risposta: il numero.

Esempio (train-5; altri 3 in `results/v5/examples_formats.md`):

```
A: There are 80/100 * 10 = 8 more purple flowers than yellow flowers.
   So in Mark's garden, there are 10 + 8 = 18 purple flowers. ...  The answer is 35.
B: There 80/100 * 10 = 8 more purple flowers than yellow flowers.
   So Mark's garden, there 10 + 8 = 18 purple flowers. ...  answer 35.
C: GIVEN yellow_flowers = 10
   purple_excess = 80 / 100 * 10 = 8
   purple_flowers = 10 + 8 = 18
   yellow_plus_purple = 10 + 18 = 28
   green_flowers = 25 / 100 * 28 = 7
   total_flowers = 28 + 7 = 35
   ANSWER = 35
D: 35
```

## 3. Dati

| | A pieno | B telegrafico | C canonico | D solo risposta |
|---|---|---|---|---|
| Validi su 7473 (train) | 7440 (99,6%) | derivato da A | 6689 (89,5%) | tutti |
| Intersezione usata | 6681 | 6681 | 6681 | 6681 |
| Token medi per soluzione (tokenizer Qwen3) | 95,4 | 83,2 (−13%) | 70,9 (−26%) | 2,3 |
| Token mediani | 88 | 77 | 66 | 2 |

**Criteri di validità:**
- risposta finale uguale a quella ufficiale;
- tutti i risultati intermedi ufficiali (`<<…=r>>`) presenti tra i numeri del testo, così i passaggi sono conservati;
- per C, grammatica e aritmetica verificate dal parser.

Fino a 3 tentativi: il primo greedy, i successivi a T = 0,7. La generazione ha richiesto 58 minuti.

**Scarti di C:** sono soprattutto problemi algebrici con incognite ("S = number_of_people…"), nomi usati dentro le espressioni e frazioni.

**Split:**
- Gli ultimi 300 problemi dell'intersezione sono il **dev**, fuori dal training di tutti i formati.
- Il training usa **6381 problemi**.
- Il test è il GSM8K test completo (1319).

## 4. Training

**Configurazione**, identica per ogni modello × formato:
- LoRA r = 16, α = 32, dropout 0,05 su q, k, v, o, gate, up, down;
- adapter in fp32 su base bf16;
- 2 epoche, batch 32, quindi **400 step per tutti**;
- AdamW, warmup del 3% e poi coseno, clip 1,0, max 640 token.

**Sweep del lr** (A, modello piccolo, seed 0, accuratezza sul dev di 300 problemi):

| lr | 1e-4 | 2e-4 | 4e-4 |
|---|---|---|---|
| Accuratezza sul dev | 0,677 | 0,677 | **0,690** |

Ho tenuto **4e-4** per tutti i run.

**Seed:** 3 sul modello piccolo e 2 sul grande, per 20 run principali.

**Tempo di training medio per run:**

| Modello | A | B | C | D |
|---|---|---|---|---|
| small | 357 s | 328 s | 323 s | 215 s |
| large | 643 s | 564 s | 521 s | 322 s |

## 5. Risultati (GSM8K test, 1319 problemi)

**Condizioni di valutazione:**
- greedy, KV cache statica, batch 32, stop su EOS;
- latenza a batch 1 su 200 problemi fissi (seed 0);
- IC 95% bootstrap sui problemi, con i seed mediati.

| Modello | Formato | Accuratezza [IC 95%] | Conformità | Token medi (mediana) | Latenza b=1 (s) | ms/token | Velocità vs A (b=1) | Risposte/s b=32 |
|---|---|---|---|---|---|---|---|---|
| 0,6B | A pieno | **56,7%** [54,2–59,1] | 99,5% | 99,9 (91) | 1,502 | 14,8 | 1,00× | 4,01 |
| 0,6B | B telegrafico | 55,3% [53,0–57,7] | 97,6% | 87,9 (80) | 1,318 | 14,9 | 1,14× | 5,02 |
| 0,6B | C canonico | 52,4% [50,0–54,8] | 91,9% | 76,1 (68) | 1,160 | 14,9 | 1,30× | 5,12 |
| 0,6B | D solo risposta | 16,4% [14,7–18,1] | 100% | 3,4 (3) | 0,055 | 15,8 | 27× | 136,5 |
| 1,7B | A pieno | **72,4%** [70,2–74,6] | 99,8% | 101,0 (93) | 1,673 | 16,6 | 1,00× | 3,73 |
| 1,7B | B telegrafico | 69,9% [67,6–72,1] | 98,0% | 87,8 (80) | 1,392 | 16,0 | 1,20× | 4,48 |
| 1,7B | C canonico | 69,1% [66,9–71,5] | 93,5% | 74,3 (68) | 1,141 | 15,5 | 1,47× | 5,05 |
| 1,7B | D solo risposta | 23,5% [21,3–25,6] | 100% | 3,4 (3) | 0,061 | 17,7 | 27× | 75,3 |

**Confronti appaiati** (McNemar esatto). La prima colonna aggrega i seed (seed s contro seed s); la seconda dà il p di ogni seed, più prudente perché i seed non sono indipendenti.

| Modello | Confronto | Δ accuratezza | p aggregato | p per seed |
|---|---|---|---|---|
| 0,6B | A − B | +1,4 | 0,071 | 0,21 / 0,59 / 0,017 |
| 0,6B | A − C | +4,3 | < 0,001 | 0,005 / 0,031 / < 0,001 |
| 0,6B | B − C | +2,9 | < 0,001 | 0,12 / 0,006 / 0,078 |
| 1,7B | A − B | +2,6 | 0,002 | 0,026 / 0,035 |
| 1,7B | A − C | +3,3 | < 0,001 | 0,002 / 0,069 |
| 1,7B | B − C | +0,7 | 0,47 | 0,30 / 1,00 |
| tutti | A, B, C − D | +36 … +49 | < 0,001 | < 0,001 |

**Grafici:** `results/v5/plots/acc_vs_tokens_small.png` e `acc_vs_tokens_large.png`, accuratezza contro token generati, un punto per formato.

**Metriche aggiuntive** (`results/v5/extra_metrics.csv`, medie sui seed):

| | A 0,6B | B 0,6B | C 0,6B | A 1,7B | B 1,7B | C 1,7B |
|---|---|---|---|---|---|---|
| Output con almeno un'uguaglianza `a op b = c` sbagliata | 9,0% | 8,4% | 7,0% | 7,4% | 6,7% | 3,5% |
| …di cui comunque corretti | 24% | 21% | 6% | 43% | 38% | 9% |
| Accuratezza se conforme | 57,0% | 55,9% | 54,8% | 72,6% | 70,2% | 70,5% |
| Accuratezza se non conforme | 0% | 33% | 25% | 0% | 52% | 50% |
| Troncati (limite di token) | 0,5% | 0,6% | 1,0% | 0,2% | 0,2% | 0,2% |

**Lettura delle metriche aggiuntive:**
- Il canonico sbaglia meno conti, ma quando ne sbaglia uno la risposta è quasi sempre sbagliata: ogni riga è un passaggio della catena, e non c'è testo che permetta di "riprendersi".
- Sull'1,7B gli output conformi di C hanno la stessa accuratezza di B (70,5% contro 70,2%). Il distacco da A viene in parte dal 6,5% di output fuori grammatica.

## 6. Analisi degli errori

**Metodo:**
- 20 errori campionati per modello × formato (seed 0), 160 in tutto.
- Li ha classificati un agente LLM separato (Sonnet), leggendo ogni output contro domanda e soluzione ufficiale.
- Ho controllato a mano due casi.
- File: `results/v5/errors_classified.csv`.

| | aritmetico | passaggio mancante | quantità confusa | piano sbagliato | formato rotto | altro | di cui da ambiguità del formato |
|---|---|---|---|---|---|---|---|
| A 0,6B | 1 | 5 | 12 | 2 | 0 | 0 | 0 |
| B 0,6B | 2 | 3 | 8 | 6 | 0 | 1 | 1 |
| C 0,6B | 1 | 3 | 7 | 7 | 1 | 1 | 3 |
| A 1,7B | 2 | 4 | 8 | 6 | 0 | 0 | 0 |
| B 1,7B | 2 | 5 | 10 | 3 | 0 | 0 | 4 |
| C 1,7B | 0 | 2 | 10 | 7 | 1 | 0 | 6 |

D: 40 su 40 sono "nessun ragionamento".

**La categoria dominante è la quantità confusa, in tutti i formati.** Prevalgono:
- percentuale applicata alla base sbagliata;
- valore unitario al posto del totale;
- somma al posto di sottrazione.

Gli errori aritmetici puri sono pochi (8 su 120).

**Errori da ambiguità del formato** (il fenomeno visto in v4): **0 in A, 5 in B, 9 in C**.

- **B: lo stripping cancella le parole che distinguono le relazioni.**
  - test-405: "1 pound each of meat and cheese" diventa "1 pound meat and 1 pound cheese… 2 pounds per sandwich". Poi i 10 pound vengono pagati due volte: 100 invece di 50.
  - test-159: "come out of hiding from under a lily pad" diventa "come out hiding under lily pad". Il modello legge una diminuzione invece di un aumento.
  - test-24: "$19.50 with 25% discount from original price" diventa "discount $19.50 25% original price".

  È lo stesso meccanismo delle collisioni di v4: togliere *of*, *from*, *with* fonde due relazioni diverse.
- **C: l'ambiguità si sposta nei nomi.**
  - **Nomi che "mentono":** in test-1245 `birds_in_jamal_photo = 1800 / 6` è in realtà la capacità di un telefono; in test-322 `boxes_jam_has = 24 / 3` sono matite per scatola.
  - **Nomi ridefiniti con significati diversi:** in test-1145 `bouquets_per_day` vale prima 7, poi 1,75, poi 4.
  - **Nomi senza unità:** in test-162 `GIVEN price_increase_per_two_months = 20` è il 20% trattato come 20 $.

  La grammatica vieta le ridefinizioni, ma il modello le produce lo stesso: è una parte del 6,5–8% di output non conformi. I casi sono più frequenti sull'1,7B, che scrive catene più lunghe.
- **Fragilità di C:** conversioni di unità, percentuali, sistemi con incognite (che il formato non può esprimere), e 2 loop di righe `FACT`.

## 7. Risposte alle domande

**Il ragionamento compresso mantiene l'accuratezza con meno token?**
- Quasi tutta, non tutta.
- Il guadagno del ragionamento rispetto a D è di 40 punti (0,6B) e 49 (1,7B).
- Conservato da B: 97% (0,6B) e 95% (1,7B), con −12% e −13% di token.
- Conservato da C: 89% e 93%, con −24% e −26% di token.
- La perdita assoluta va da 1,4 a 4,3 punti. È significativa in tutti i casi tranne A−B sullo 0,6B (p = 0,07 aggregato, non concorde tra i seed).

**Di quanto è più veloce?**
- Il costo per token generato resta costante (~15 ms sullo 0,6B, ~16 ms sull'1,7B), quindi la velocità segue il numero di token.
- Latenza a batch 1:

  | Formato | 0,6B | 1,7B |
  |---|---|---|
  | B | 1,14× (1,50 → 1,32 s) | 1,20× (1,67 → 1,39 s) |
  | C | 1,30× (→ 1,16 s) | 1,47× (→ 1,14 s) |

- A batch 32 le risposte al secondo crescono di 1,25× e 1,20× per B, 1,28× e 1,35× per C.
- Anche il training è più rapido, con meno token per esempio: C impiega 0,90× (0,6B) e 0,81× (1,7B) il tempo di A.

**Canonico o telegrafico?**
- Sull'**1,7B il canonico è migliore**: stessa accuratezza di B (−0,7 punti, p = 0,47) con il 15% di token in meno e 1,22× più veloce a batch 1.
- Sullo **0,6B è un compromesso**: C ha −13% token rispetto a B ma −2,9 punti.
- Perché:
  1. La forma regolare comprime molto di più senza togliere le parole a caso: i 12 punti di token in più che C toglie rispetto a B vengono dalla sintassi, non dalle relazioni.
  2. Ogni passaggio è verificabile e non c'è "prosa" in cui confondersi; infatti C sbaglia meno conti.
  3. Ma il modello deve **imparare una lingua nuova** con 6,4k esempi. Il modello piccolo la rispetta meno (8% di output non conformi) e mette nei nomi delle quantità informazioni sbagliate o vaghe, che poi rilegge.
  4. Il modello più grande impara meglio la grammatica e recupera la differenza con B.
- In proporzione al costo, C è il formato con il miglior rapporto accuratezza/token su entrambi i modelli. È l'unico che sull'1,7B sta sulla frontiera insieme ad A.

## 8. Confronto con v4

| Conclusione di v4 (dati sintetici, modelli da zero) | In v5 (GSM8K, modelli pre-addestrati + LoRA) |
|---|---|
| Togliere parole alla cieca non aiuta e crea ambiguità (collisioni tra fatti) | **Confermato.** B perde 1,4–2,6 punti e gli errori da ambiguità sono proprio preposizioni perse (*of*, *from*, *with*) che fondono relazioni. Riduce solo il 12–13% dei token, perché i numeri e i conti restano |
| La forma canonica e regolare aiuta l'apprendimento (−9/−19% parametri al 50%) | **Non si traduce in accuratezza.** Su modelli pre-addestrati il linguaggio naturale resta il formato in cui il modello "sa" ragionare: A è il più accurato. La regolarità aiuta il modello grande a pareggiare B con meno token, non a battere A |
| L'ambiguità nasce dalle collisioni tra template | Nel canonico l'ambiguità **si sposta nei nomi** delle quantità (vaghi, sbagliati, ridefiniti): un nuovo canale di ambiguità che in v4 non esisteva, perché lì i nomi erano fissi |
| Il guadagno pratico della compressione è nel numero di token (tempo di training 0,55–0,66×) | **Confermato e misurato in velocità:** −24/−26% di token si traducono in latenza 1,30–1,47× migliore e throughput 1,28–1,35×. Anche il training è più rapido (0,81–0,90×) |

## 9. Limiti e deviazioni

- **Generatore locale invece dell'API:**
  - Non c'era la chiave; ho usato Qwen3-4B-Instruct (Q8) con llama.cpp, scelta approvata.
  - La qualità di A e C è quella di un modello da 4B, filtrata dai controlli. Un generatore più forte potrebbe dare nomi canonici migliori e meno scarti.
  - Lo stesso generatore, della famiglia Qwen3, scrive i dati per i modelli Qwen3 che vengono addestrati.
- **Sottoinsieme di training:** l'intersezione dei validi (89%) esclude soprattutto problemi algebrici che C non sa esprimere. È uguale per tutti i formati, ma il test li contiene, quindi C, che non ha una forma per le incognite, può essere svantaggiato su quei problemi.
- **Normalizzazione:** gli output del generatore sono stati normalizzati in modo deterministico prima della verifica (sezione 2). Gli output dei modelli addestrati sono valutati senza normalizzazione.
- **lr:**
  - Lo sweep era richiesto solo su A e sul modello piccolo. Il migliore, 4e-4, è sul bordo della griglia, e le differenze (67,7–69,0% su 300 problemi) sono nel rumore.
  - Lo stesso lr per tutti i formati e i modelli potrebbe favorire A.
  - Dev di 300 problemi tolti al training di tutti, quindi 6381 esempi.
- **Seed:** 3 sullo 0,6B e 2 sull'1,7B. Il McNemar aggregato tratta i seed come indipendenti ed è ottimista; per questo riporto anche il p per seed.
- **Latenza e throughput:**
  - Misurati con `generate` di transformers, KV cache statica, compilazione disattivata. Con transformers 5.22 su Windows la compilazione automatica della cache statica fallisce (`NameError: torch`).
  - I ~15 ms per token sono dominati dall'overhead per passo, non dal modello. Con motori ottimizzati i tempi assoluti scendono, ma il rapporto tra formati, che segue i token, dovrebbe restare.
  - Il throughput a batch 32 usa l'ordine del test, senza ordinamento per lunghezza.
- **Modifiche tecniche dopo il quick run:**
  - Logit calcolati solo sulle posizioni con target: è la stessa loss, ma con il vocabolario da 152k i logit pieni non stavano in memoria.
  - Micro-batch 6 (0,6B) e 4 (1,7B) con accumulo fino a 32: oltre ~10 GB Windows riversava in RAM.
  - Quick run valutato su 128 problemi del test e latenza su 20.
- **Analisi degli errori:** 20 errori per cella, classificati da un LLM e non da annotatori umani. I conteggi delle categorie sono indicativi. Il controllo automatico delle uguaglianze (`extra_metrics.csv`) è approssimato da un'espressione regolare.
- **Possibile contaminazione:** GSM8K potrebbe essere nel pre-training dei Qwen3 base, nello stesso modo per tutti i formati. Due "errori" su 160 sono discutibili: test-823 ha la soluzione ufficiale sbagliata, test-1227 una domanda ambigua.
- **Modelli:** non ho usato i Qwen3.5 base, più recenti (sezione 1).

## File

| Contenuto | Percorso |
|---|---|
| Dati per formato | `v5/data/sft/*.jsonl` |
| Riscritture con esito della verifica | `v5/data/rewrites/{A,C}.jsonl` |
| Statistiche ed esempi | `results/v5/data_stats.csv`, `results/v5/examples_formats.md` |
| Run (inclusi sweep e quick) | `results/v5/runs.csv` |
| Riepilogo e confronti | `results/v5/summary.csv`, `results/v5/pairwise.csv`, `results/v5/mcnemar_per_seed.csv`, `results/v5/extra_metrics.csv` |
| Latenza | `results/v5/latency.csv`, più per problema in `results/v5/outputs/*_latency.jsonl` |
| Output di ogni run sul test | `results/v5/outputs/*_test.jsonl` |
| Errori | `results/v5/errors_sample.jsonl`, `results/v5/errors_classified.csv` |
| Grafici | `results/v5/plots/acc_vs_tokens_{small,large}.png` |
