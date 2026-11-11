# Esperimento v2 — compressione linguistica e parametri necessari

> Stato: **piano completo** (Fasi A, B, C e tutti i blocchi extra), concluso l'8/11/2026 alle 00:03.
> Tabelle grezze rigenerate automaticamente: `results/v2/summary_v2.md`; CSV in `results/v2/*.csv`.

## 1. Sintesi

**Monolingua (Fase A, pari passi, E200, 20 000 persone, 2 seed).** Lo spogliamento del testo riduce
i parametri necessari solo di poco: il formato **telegrafico** richiede **~0,89×** i parametri
non-embedding dell'italiano pieno sia al 50% sia al 90% di recall; l'**interlingua nativa** ~0,91×
al 50% e ~1,02× al 90%. Il testo arricchito (**italiano_ricco**, 1,66× i token) ne richiede
~1,1×. Le differenze sono dell'ordine del 10%, a fronte di differenze di lunghezza del testo del
30–65%: **i dati supportano H0** (la capacità va ai fatti, non alla forma), con un piccolo effetto
residuo nella direzione di H1. Il risultato conferma il controllo a pari passi della v1 (0,89–0,91
per telegrafico, ~1,0–1,05 per interlingua).

**Multilingua (Fase B, 3 seed).** Un modello che impara metà delle persone in italiano e metà in
inglese richiede gli **stessi parametri** del monolingua per ricordarle nella lingua in cui le ha
viste (0,94× al 50%, 0,96× al 90%; 10% di persone "ponte" in entrambe le lingue: 0,96× / 0,95×).
Però **non trasferisce i fatti tra lingue**: chiesti nell'altra lingua, gli attributi tradotti
(corso di laurea, professione) restano all'1–3% anche a 4M parametri; passano solo i valori con la
stessa forma superficiale nelle due lingue (università, azienda, in parte città). Il ponte raddoppia
la recall cross-lingua (0,25 → 0,47 a 4M) ma solo sui valori invariati. I fatti vengono memorizzati
come associazioni tra stringhe, non come contenuto indipendente dalla lingua: anche qui **H0**.

**Costi end-to-end (Fase C).** La compressione fa risparmiare sul **training**: −38% di FLOP per
arrivare al 90% col telegrafico, −23% con l'interlingua. In **risposta** a una domanda fattuale la
pipeline compressa (stripper → core compresso → traduttore) è **più lenta** del core pieno da solo:
~6× con batch=1, ~4× con batch=64. La risposta è il valore, uguale in tutti i formati, e i due
modelli di traduzione aggiungono un costo fisso. La compressione diventa **più veloce** solo quando
il modello genera molto testo. Nello scenario "ragionamento interno" (200 token intermedi + 20 di
risposta) il core telegrafico impiega ~0,70× il tempo del pieno se scrive direttamente la risposta
finale in italiano pieno, ~0,82× con stripper e traduttore. Vedi §7.

## 2. Cosa cambia rispetto alla v1

| # | Correzione | Implementazione |
|---|---|---|
| 1 | Protocollo principale a **pari passi** | 512 biografie/step per tutti i formati, steps = E·N/512; pari esposizioni come blocco secondario (A_esp) |
| 2 | Sweep esposizioni | E200 nel main; E100 e E500 come blocchi extra |
| 3 | 3 learning rate | 0,5×, 1×, 2× su 3 taglie (0,20M, 0,38M, 0,79M), si tiene il migliore |
| 4 | Interlingua nativa | date ISO (`1987-03-12`), entità con underscore (`Mario_Rossi`), confronto dei fatti normalizzato |
| 5 | Cifre singole | pre-tokenizer `Digits(individual_digits=True)` in tutti i tokenizer |
| 6 | Griglia taglie | 8 taglie log-spaziate 0,15M–1,49M + 4M, 2 seed (4M: seed 0) |
| 7 | Parametri totali e non-embedding | entrambi riportati; tokenizer condiviso pieno+interlingua come blocco extra |
| 8 | italiano_ricco | ≥10 template per attributo, articoli, subordinate, 30 frasi riempitive |
| 9 | bit/parametro | Σ acc_attr × entropia_attr × N / params_nonemb |
| 10 | Throughput a regime | token/s e fatti/s esclusi generazione cache e primi 50 step |

## 3. Learning rate scelto

| Taglia | pieno | telegrafico | interlingua | ricco |
|---|---|---|---|---|
| 0,20M | 2× | 2× | 2× | 2× |
| 0,38M | 2× | 2× | 2× | 2× |
| 0,79M | 1× | 1× | 1× | 2× |

Le taglie non provate usano il lr della taglia provata più vicina (in log-parametri). A 0,79M la
differenza tra 1× e 2× è ≤0,02 di accuratezza; a 0,20–0,38M il 2× vince ovunque ed è al bordo della
griglia (vedi Limiti). La scelta è stata uniforme tra i formati, quindi non favorisce nessuno.

## 4. Fase A — risultati principali (pari passi, E200)

### Accuratezza media (exact match sui 6 attributi, 5000 persone, media su 2 seed)

| Taglia (non-emb) | pieno | telegrafico | interlingua | ricco |
|---|---|---|---|---|
| 0,15M | 0,266 | 0,385 | 0,381 | 0,224 |
| 0,20M | 0,434 | 0,432 | 0,442 | 0,257 |
| 0,31M | 0,581 | 0,771 | 0,677 | 0,588 |
| 0,38M | 0,674 | 0,805 | 0,710 | 0,588 |
| 0,60M | 0,953 | 0,972 | 0,930 | 0,902 |
| 0,79M | 0,997 | 1,000 | 0,993 | 0,960 |
| 0,99M | 1,000 | 1,000 | 0,999 | 0,984 |
| 1,49M | 1,000 | 1,000 | 1,000 | 0,999 |
| 4M (seed 0) | 1,000 | 1,000 | 1,000 | 1,000 |

Grafico: `results/v2/acc_vs_params_passi.png`.

### Parametri al 50% e al 90% di recall (interpolazione log, curva media)

| Formato | token/bio | 50% non-emb | 90% non-emb | rapporto 50% (non-emb / totali) | rapporto 90% (non-emb / totali) |
|---|---|---|---|---|---|
| italiano_pieno | 50,8 | 244k | 559k | 1,00 / 1,00 | 1,00 / 1,00 |
| telegrafico | 34,3 | 219k | 501k | **0,89 / 0,91** | **0,90 / 0,90** |
| interlingua | 38,1 | 223k | 573k | **0,91 / 0,93** | **1,02 / 1,02** |
| italiano_ricco | 84,2 | 277k | 606k | **1,13 / 1,15** | **1,09 / 1,12** |

Stabilità tra seed (rapporti non-emb calcolati per singolo seed, seed 0 / seed 1):
telegrafico 0,93/0,86 (50%) e 0,90/0,90 (90%); interlingua 0,96/0,85 e 0,99/1,05; ricco 1,16/1,11
e 1,12/1,07. Il segno dell'effetto è stabile per telegrafico (<1) e ricco (>1); per l'interlingua al
90% non lo è.

### Confronto v1 → v2 (rapporti non-emb vs italiano_pieno)

| Formato | v1 pari esposizioni 50% / 90% | v1 pari passi 50% / 90% | **v2 pari passi 50% / 90%** |
|---|---|---|---|
| telegrafico | 1,18 / 1,45 | 0,91 / 0,89 | **0,89 / 0,90** |
| interlingua | 1,58 / 2,01 | 1,01 / 1,05 | **0,91 / 1,02** |

Il "vantaggio" apparente dell'italiano pieno nella v1 a pari esposizioni era dovuto al compute
(il pieno faceva ~1,5–1,75× gli step); a pari passi, con lr ottimizzato, tokenizer a cifre
singole e interlingua nativa, i formati compressi sono leggermente favoriti ma di poco.

### Bit memorizzati per parametro (E200)

| Taglia | pieno | telegrafico | interlingua | ricco |
|---|---|---|---|---|
| 0,15M | 1,54 | **2,21** | **2,21** | 1,30 |
| 0,20M | 1,88 | 1,87 | 1,92 | 1,13 |
| 0,31M | 1,62 | 2,12 | 1,87 | 1,63 |
| 0,60M | 1,53 | 1,57 | 1,47 | 1,47 |
| 0,99M | 1,00 | 1,00 | 1,00 | 0,98 |

Il massimo (~2,2 bit/parametro, a 0,15M per telegrafico e interlingua; 1,9 per il pieno a 0,20M;
1,6 per il ricco a 0,31M) è vicino ai ~2 bit/parametro riportati in letteratura per la capacità di
memorizzazione dei transformer. Sopra ~0,8M l'accuratezza satura (bit totali fissi ≈ 1,0 Mbit) e il
rapporto scende come 1/params: lì la metrica misura solo il sovradimensionamento.

### Throughput a regime (A_main, media sulle taglie)

| Formato | token/s | fatti/s | tempo di training medio per run |
|---|---|---|---|
| italiano_pieno | 1,23M | 143k | 178 s |
| telegrafico | 1,07M | 181k | 141 s |
| interlingua | 1,07M | 164k | 156 s |
| italiano_ricco | 1,37M | 97k | 275 s |

A pari passi i formati compressi costano meno compute per fatto (telegrafico ~0,79× il tempo del
pieno, ricco ~1,55×): il guadagno della compressione è soprattutto in **costo di training**, non
in **parametri**.

## 5. Ablazioni (A_abl)

Stesse condizioni del main (pari passi, E200, lr scelto), 6 taglie 0,15M–0,79M, 2 seed.

| Taglia | pieno | pieno senza cifre singole | interlingua nativa | interlingua codifica v1 |
|---|---|---|---|---|
| 0,15M | 0,266 | 0,213 | 0,381 | 0,382 |
| 0,20M | 0,434 | 0,310 | 0,442 | 0,484 |
| 0,31M | 0,581 | 0,547 | 0,677 | 0,759 |
| 0,38M | 0,674 | 0,628 | 0,710 | 0,794 |
| 0,60M | 0,953 | 0,976 | 0,930 | 0,990 |
| 0,79M | 0,997 | 0,999 | 0,993 | 1,000 |
| token/bio | 50,8 | 45,1 | 38,1 | 30,3 |
| **rapporto 50% / 90%** vs pieno | 1 / 1 | **1,17 / 0,99** | 0,91 / 1,02 | **0,84 / 0,89** |

- **Cifre singole:** senza cifre singole il pieno è più lento a salire (1,17× i parametri al 50%) ma
  arriva al 90% con gli stessi parametri. Le date sono l'attributo più costoso (14,3 bit) e
  vengono imparate per ultime in tutti i formati (≤2% fino a 0,38M, poi 70–99% tra 0,6M e 0,8M);
  con anni come token unici il modello piccolo ne ricorda qualcuna in più (10–36% a 0,20–0,31M).
- **Interlingua:** la codifica v1 (più corta, 30 token/bio) è *migliore* di quella nativa (date ISO
  e underscore, 38 token/bio): 0,84× / 0,89× contro 0,91× / 1,02×. La "normalizzazione" nativa non
  aiuta: allunga la sequenza e spezza i nomi composti in token meno riusabili. Con la codifica v1
  l'interlingua è il formato più efficiente in parametri, ma il guadagno resta ≤16%.

## 6. Fase B — multilingua

**Disegno.** 20 000 persone, ognuna assegnata a una lingua (metà italiano, metà inglese), testo pieno,
tokenizer BPE unico (4096) sulle due lingue, pari passi (7812 step × 512 biografie), E200, lr scelto per
italiano_pieno, 9 taglie × 3 seed (anche 4M).
- **multi_pieno (B1):** ogni persona solo nella sua lingua.
- **multi_bridge (B3):** in più il 10% delle persone ("ponte") compare in entrambe le lingue.
- **Recall in-lingua:** domanda nella lingua di training della persona. **Cross-lingua:** persona non
  ponte interrogata nell'altra lingua; *strict* = risposta corretta nella lingua della domanda,
  *lenient* = corretta in una qualsiasi delle due lingue (differisce di <0,01: il modello non risponde
  "nella lingua sbagliata", semplicemente non sa).

### Recall in-lingua e cross-lingua (media su 3 seed, ± dev. std)

| Taglia | italiano_pieno (A) | multi_pieno in | multi_bridge in | multi_pieno cross | multi_bridge cross |
|---|---|---|---|---|---|
| 0,15M | 0,266 | 0,296 | 0,275 | 0,050 | 0,113 |
| 0,20M | 0,434 | 0,420 | 0,402 | 0,055 | 0,163 |
| 0,31M | 0,581 | 0,682 | 0,675 | 0,110 | 0,280 |
| 0,38M | 0,674 | 0,732 | 0,737 | 0,091 | 0,272 |
| 0,60M | 0,953 | 0,967 | 0,968 | 0,149 | 0,356 |
| 0,79M | 0,997 | 0,995 | 0,992 | 0,157 | 0,347 |
| 0,99M | 1,000 | 0,999 | 0,999 | 0,189 | 0,383 |
| 1,49M | 1,000 | 1,000 | 1,000 | 0,207 | 0,411 |
| 4M | 1,000 | 1,000 | 1,000 | 0,253 ± 0,008 | 0,465 ± 0,021 |

Dev. std tra seed della recall in-lingua ≤0,018 a tutte le taglie. Grafici:
`results/v2/multilingua.png`, `results/v2/cross_vs_size.png`.

### Parametri al 50% e al 90% (recall in-lingua)

| Corpus | 50% non-emb | 90% non-emb | rapporto vs italiano_pieno 50% / 90% | per seed 50% (s0/s1/s2) | per seed 90% |
|---|---|---|---|---|---|
| italiano_pieno (A) | 244k | 559k | 1 / 1 | – | – |
| multi_pieno | 229k | 536k | **0,94 / 0,96** | 0,95 / 0,92 / 0,94 | 0,95 / 0,95 / 0,97 |
| multi_bridge | 234k | 533k | **0,96 / 0,95** | 0,97 / 0,95 / 0,96 | 0,94 / 0,96 / 0,96 |

In parametri totali i rapporti sono 0,99 / 0,99 (multi_pieno) e 1,01 / 0,99 (multi_bridge). La
recall cross-lingua **non raggiunge il 50% a nessuna taglia** (massimo 0,47 a 4M con ponte), quindi
per essa non esiste una soglia.

### Cross-lingua per attributo (4M)

| Attributo | stessa forma nelle due lingue? | multi_pieno | multi_bridge |
|---|---|---|---|
| università | sì ("Scuola Normale Superiore") | 0,90 | 0,98 |
| azienda | sì ("Generali") | 0,32 | 0,80 |
| luogo di nascita | sì ("Viareggio") | 0,22 | 0,66 |
| data di nascita | in parte ("1 dicembre" / "1 December") | 0,05 | 0,30 |
| corso | no ("Storia medievale" / "Medieval history") | 0,02 | 0,03 |
| professione | no ("oculista" / "ophthalmologist") | 0,01 | 0,02 |

Il caso a priori è <1% per tutti gli attributi (≥120 valori ciascuno).

**Lettura.**
1. **Imparare due lingue non costa parametri in più** a pari numero di fatti: ogni fatto è memorizzato
   una volta, nella lingua in cui compare. Il leggero vantaggio (0,94–0,96×) è dentro la variabilità
   vista in Fase A tra seed.
2. **Il fatto non è separato dalla sua forma.** Se il valore va tradotto, il modello non lo
   recupera nell'altra lingua (1–3% anche a 4M). Recupera solo valori che può copiare identici:
   università e aziende molto, città e date meno. La recall cross-lingua cresce lentamente con la
   taglia (0,05 → 0,25 senza ponte) e solo grazie a questi valori.
3. **Il 10% di persone ponte** raddoppia la recall cross-lingua (0,25 → 0,47 a 4M) senza peggiorare
   quella in-lingua, ma non sblocca gli attributi tradotti (corso e professione restano ≤3%). Il ponte
   insegna al modello a riusare un nome nell'altra lingua, non a tradurre il valore.

## 7. Costi end-to-end (Fase C)

**Condizioni di misura.**
- GPU dedicata, senza altri training in corso.
- Core di A_main (seed 0) a 0,60M, 1,49M e 4M non-emb.
- bf16, generazione greedy con KV cache, 20 chiamate di warm-up.
- 1000 domande di valutazione "in" (le prime 1000 del set).
- Con batch=64 i tempi sono **per lotto** di 64 domande, cioè la latenza vista da un utente dentro un
  lotto; il throughput è in risposte/s.

### 7.1 Training

| Taglia | Formato | token/s a regime | tempo (s) | FLOP totali | acc | FLOP per fatto memorizzato |
|---|---|---|---|---|---|---|
| 0,60M | italiano_pieno | 1,19M | 174 | 1,30·10¹⁵ | 0,924 | 1,17·10¹⁰ |
| 0,60M | telegrafico | 1,07M | 132 | 8,78·10¹⁴ | 0,950 | **7,70·10⁹** (0,66×) |
| 0,60M | interlingua | 1,05M | 148 | 9,78·10¹⁴ | 0,923 | 8,83·10⁹ (0,76×) |
| 1,49M | italiano_pieno | 1,02M | 203 | 2,71·10¹⁵ | 1,000 | 2,26·10¹⁰ |
| 1,49M | telegrafico | 0,75M | 189 | 1,84·10¹⁵ | 1,000 | **1,53·10¹⁰** (0,68×) |
| 1,49M | interlingua | 0,82M | 192 | 2,04·10¹⁵ | 1,000 | 1,70·10¹⁰ (0,75×) |
| 4M | italiano_pieno | 0,62M | 333 | 6,36·10¹⁵ | 1,000 | 5,30·10¹⁰ |
| 4M | telegrafico | 0,63M | 224 | 4,32·10¹⁵ | 1,000 | **3,60·10¹⁰** (0,68×) |
| 4M | interlingua | 0,58M | 269 | 4,80·10¹⁵ | 1,000 | 4,00·10¹⁰ (0,75×) |

- FLOP = 6 × parametri totali × token di training.
- Fatti memorizzati = acc × 120 000 (6 attributi × 20 000 persone).

**Alla soglia del 90%** (parametri totali interpolati):

| Formato | parametri totali | FLOP | FLOP per fatto | rapporto vs pieno |
|---|---|---|---|---|
| italiano_pieno | 967k | 1,20·10¹⁵ | 1,11·10¹⁰ | 1 |
| telegrafico | 872k | 7,39·10¹⁴ | 6,84·10⁹ | **0,62×** |
| interlingua | 988k | 9,27·10¹⁴ | 8,58·10⁹ | **0,77×** |

In tempo reale il risparmio è minore: il telegrafico richiede 0,67–0,93× il tempo del pieno. A queste
taglie la GPU è limitata dall'overhead e le sequenze corte la sfruttano peggio.

### 7.2 Risposta del core (domanda fattuale)

| Formato | Taglia | token prompt | token risposta | latenza b=1 (ms) | TTFT b=1 (ms) | ms/token decodifica b=1 | risposte/s b=64 | FLOP/risposta | acc |
|---|---|---|---|---|---|---|---|---|---|
| italiano_pieno | 0,60M | 6,0 | 3,08 | 2,38 | 0,91 | 0,73 | 8 373 | 1,90·10⁷ | 0,928 |
| italiano_pieno | 1,49M | 6,0 | 3,08 | 2,51 | 0,96 | 0,77 | 8 149 | 3,97·10⁷ | 1,000 |
| italiano_pieno | 4M | 6,0 | 3,08 | 2,51 | 0,99 | 0,76 | 8 251 | 9,31·10⁷ | 1,000 |
| telegrafico | 0,60M | 4,3 | 3,08 | 2,40 | 0,94 | 0,72 | 9 579 | 1,54·10⁷ | 0,942 |
| telegrafico | 1,49M | 4,3 | 3,08 | 2,55 | 1,01 | 0,77 | 8 111 | 3,22·10⁷ | 1,000 |
| telegrafico | 4M | 4,3 | 3,08 | 2,50 | 1,01 | 0,74 | 7 940 | 7,57·10⁷ | 1,000 |
| interlingua | 0,60M | 5,7 | 4,25 | 3,46 | 1,03 | 0,77 | 6 092 | 2,08·10⁷ | 0,935 |
| interlingua | 1,49M | 5,7 | 4,25 | 3,31 | 1,00 | 0,72 | 5 493 | 4,34·10⁷ | 1,000 |
| interlingua | 4M | 5,7 | 4,25 | 3,25 | 0,97 | 0,72 | 5 229 | 1,02·10⁸ | 1,000 |

- **Il tempo per token (~0,75 ms) non dipende dalla taglia** tra 0,6M e 4M. A questa scala conta il
  numero di passi di generazione, non i FLOP, perché domina il costo di lancio dei kernel. I FLOP
  crescono con la taglia ma non si vedono nel tempo.
- **Telegrafico:** accorcia la domanda (4,3 contro 6,0 token), ma la risposta è lo stesso valore.
  Latenza uguale al pieno, −19% di FLOP.
- **Interlingua:** scrive risposte più lunghe (date ISO a cifre singole, 4,25 token) ed è **~35% più
  lenta**.

### 7.3 Stripper e traduttore (~1M non-emb, decoder-only con prompt)

| Modello | Formato | training | in → out token | exact match | latenza b=1 | risposte/s b=64 | FLOP/richiesta |
|---|---|---|---|---|---|---|---|
| stripper (pieno → compresso) | telegrafico | 48 s, 1,7·10¹⁴ FLOP | 7,0 → 3,3 | frase 1,000 | 4,0 ms | 12 582 | 3,6·10⁷ |
| stripper | interlingua | 49 s, 2,0·10¹⁴ FLOP | 7,0 → 4,7 | frase 1,000 | 5,2 ms | 6 393 | 4,1·10⁷ |
| traduttore (compresso → pieno) | telegrafico | 64 s, 2,8·10¹⁴ FLOP | 8,4 → 8,1 | frase 1,000, valori 1,000 | 7,9 ms | 5 266 | 5,5·10⁷ |
| traduttore | interlingua | 66 s, 3,4·10¹⁴ FLOP | 11,3 → 8,4 | frase 1,000, valori 1,000 | 8,9 ms | 3 503 | 6,7·10⁷ |

Il traduttore è lo stadio più costoso: deve scrivere la frase piena (~8 token) partendo dalla domanda
compressa e dal valore.

### 7.4 Pipeline end-to-end vs core pieno (domanda fattuale)

| Formato | Taglia | batch | stripper | core compresso | traduttore | **pipeline (ms)** | core pieno (ms) | **pipeline / pieno** | FLOP pipeline / pieno | acc pipeline / pieno |
|---|---|---|---|---|---|---|---|---|---|---|
| telegrafico | 0,60M | 1 | 4,0 | 2,5 | 8,4 | **14,9** | 2,4 | **6,1×** | 5,6× | 0,942 / 0,928 |
| telegrafico | 1,49M | 1 | 3,8 | 2,6 | 8,1 | **14,5** | 2,4 | **6,1×** | 3,1× | 1,000 / 1,000 |
| telegrafico | 4M | 1 | 4,5 | 2,3 | 8,2 | **15,1** | 2,5 | **6,1×** | 1,8× | 1,000 / 1,000 |
| telegrafico | 0,60M | 64 | 4,4 | 5,8 | 11,6 | **21,8** | 5,4 | **4,1×** | 4,4× | 0,942 / 0,927 |
| telegrafico | 1,49M | 64 | 4,4 | 6,9 | 9,2 | **20,5** | 5,7 | **3,6×** | 2,5× | 1,000 / 1,000 |
| telegrafico | 4M | 64 | 6,5 | 7,2 | 9,1 | **22,9** | 7,0 | **3,3×** | 1,6× | 1,000 / 1,000 |
| interlingua | 0,60M | 1 | 5,2 | 3,2 | 8,6 | **17,0** | 2,6 | **6,5×** | 6,8× | 0,935 / 0,928 |
| interlingua | 1,49M | 1 | 4,9 | 3,4 | 8,6 | **16,9** | 2,3 | **7,4×** | 3,8× | 1,000 / 1,000 |
| interlingua | 4M | 1 | 5,2 | 3,7 | 8,8 | **17,7** | 2,9 | **6,1×** | 2,3× | 1,000 / 1,000 |
| interlingua | 0,60M | 64 | 7,4 | 10,6 | 11,2 | **29,2** | 6,4 | **4,6×** | 5,5× | 0,936 / 0,927 |
| interlingua | 1,49M | 64 | 5,7 | 8,7 | 9,5 | **23,9** | 6,4 | **3,7×** | 3,3× | 1,000 / 1,000 |
| interlingua | 4M | 64 | 9,0 | 15,5 | 10,7 | **35,2** | 9,0 | **3,9×** | 2,2× | 1,000 / 1,000 |

- Tempi in ms per richiesta (b=1) o per lotto da 64 (b=64).
- Con il core più grande il rapporto in FLOP scende, perché gli stadi da 1M pesano meno su un core da
  4M.
- In tempo il rapporto resta ~6×, perché ogni stadio paga il proprio overhead fisso per passo di
  generazione.

### 7.5 Scenario "ragionamento interno" (200 token intermedi + 20 di risposta finale)

**È una stima analitica.** I core hanno un contesto pari alla biografia più lunga (~40–90 token) e
non possono generare 220 token.
- **Tempo** = TTFT misurato + (token − 1) × ms/token misurati (§7.2).
- **FLOP:** formula della KV cache (`src/v2/kvgen.py`).
- **Token compressi** = 200 × r, con r = rapporto di lunghezza misurato sulle biografie: telegrafico
  0,675, interlingua 0,750, interlingua codifica v1 0,596. Per quest'ultima uso i tempi per token
  dell'interlingua nativa.

Due varianti:
- **A — core compresso, risposta finale piena diretta:** 200·r token intermedi compressi + 20 token
  pieni scritti dallo stesso modello. Presuppone un core addestrato a farlo; non misurato.
- **B — pipeline completa:** stripper sulla domanda + core che genera tutto compresso (220·r token) +
  traduttore che scrive i 20 token finali in pieno.

| Formato | Taglia | tutto pieno b=1 (ms) | **A b=1** | **B b=1** | A b=64 | B b=64 | FLOP A / pieno | FLOP B / pieno |
|---|---|---|---|---|---|---|---|---|
| telegrafico | 0,60M | 160 | 112 (**0,70×**) | 133 (**0,83×**) | 0,58× | 0,70× | 0,68× | 0,95× |
| telegrafico | 1,49M | 168 | 119 (**0,71×**) | 139 (**0,83×**) | 0,67× | 0,78× | 0,69× | 0,80× |
| telegrafico | 4M | 166 | 115 (**0,69×**) | 136 (**0,82×**) | 0,71× | 0,82× | 0,69× | 0,73× |
| interlingua | 0,60M | 160 | 130 (0,81×) | 154 (0,96×) | 0,66× | 0,82× | 0,76× | 1,06× |
| interlingua | 1,49M | 168 | 123 (0,73×) | 148 (0,88×) | 0,71× | 0,86× | 0,77× | 0,90× |
| interlingua | 4M | 166 | 122 (0,73×) | 146 (0,88×) | 0,77× | 0,93× | 0,77× | 0,82× |
| interlingua cod. v1 | 0,60M | 160 | 107 (0,67×) | 128 (0,80×) | 0,54× | 0,69× | 0,62× | 0,88× |
| interlingua cod. v1 | 1,49M | 168 | 101 (0,60×) | 123 (0,73×) | 0,58× | 0,72× | 0,63× | 0,74× |
| interlingua cod. v1 | 4M | 166 | 100 (0,60×) | 122 (0,73×) | 0,63× | 0,78× | 0,63× | 0,66× |

**Punto di pareggio della pipeline B** (telegrafico, b=1):
- stripper e traduttore costano ~25 ms fissi;
- ogni token pieno risparmiato vale (1 − 0,675) × 0,75 ≈ 0,24 ms;
- la pipeline conviene quando la risposta piena supererebbe **~100 token generati**.

La variante A non ha costi fissi: conviene sempre, in proporzione ai token risparmiati.

### 7.6 Risposta: la pipeline compressa è più veloce del pieno?

- **Domanda fattuale con risposta breve (il compito dell'esperimento): NO.**
  - La pipeline è 6–7× più lenta con batch=1 e 3,3–4,6× con batch=64.
  - Il core compresso da solo è veloce quanto il pieno (telegrafico) o più lento (interlingua, +35%),
    perché la risposta è il valore e non si comprime.
- **Generazione lunga ("ragionamento interno"): SÌ, stimato.** Con 220 token generati:
  - telegrafico: ~0,70× il tempo del pieno se il core scrive direttamente la risposta finale piena,
    ~0,82× con stripper e traduttore;
  - interlingua con codifica v1: 0,60× e 0,73×;
  - il guadagno è proporzionale ai token risparmiati e scompare sotto ~100 token generati se servono
    stripper e traduttore.
- **Il risparmio più sicuro resta nel training:** −38% di FLOP per arrivare al 90% col telegrafico.

## 8. Blocchi extra

Ordine: tokenizer condiviso → E100 → E500 → pari esposizioni. Stesse condizioni del main (pari
passi, lr scelto), 6 taglie 0,15M–0,79M, 2 seed.

### 8.1 Tokenizer condiviso (A_shared)

Un solo BPE da 4096 token addestrato sull'unione di italiano_pieno e interlingua, usato da entrambi.
Così l'embedding e il vocabolario sono identici e il confronto non dipende dal tokenizer.

| Taglia | pieno (proprio) | pieno (condiviso) | interlingua (propria) | interlingua (condiviso) |
|---|---|---|---|---|
| 0,15M | 0,266 | 0,291 | 0,381 | 0,388 |
| 0,20M | 0,434 | 0,439 | 0,442 | 0,450 |
| 0,31M | 0,581 | 0,672 | 0,677 | 0,687 |
| 0,38M | 0,674 | 0,706 | 0,710 | 0,751 |
| 0,60M | 0,953 | 0,961 | 0,930 | **0,837** |
| 0,79M | 0,997 | 0,998 | 0,993 | 0,955 |
| token/bio | 50,8 | 51,4 | 38,1 | 38,6 |

**Soglie, interlingua vs pieno con lo stesso tokenizer:**

| Soglia | non-emb (pieno → interlingua) | rapporto non-emb | rapporto totali | per seed (s0 / s1) | tokenizer propri (§4) |
|---|---|---|---|---|---|
| 50% | 224k → 220k | **0,98** | 0,98 | 0,98 / 0,97 | 0,91 |
| 90% | 547k → 701k | **1,28** | 1,23 | 1,35 / 1,23 | 1,02 |

- **Pieno:** il tokenizer condiviso non lo cambia, né in parametri al 90% (547k contro 559k) né in
  lunghezza.
- **Interlingua:** sotto il 50% è identica. Al 90% richiede ~28% di parametri in più, perché impara
  le **date** più tardi: a 0,60M solo il 20% di date corrette, contro il 69% con il tokenizer proprio.
  Gli altri cinque attributi sono invariati.
- **Causa:** non trovata. Il tokenizer non è la spiegazione diretta: le date ISO sono tokenizzate in
  modo identico (cifre singole e trattini) dai due tokenizer, e cambia solo qualche nome spezzato in
  più (es. "Val"+"le"). Lo spostamento è stabile sui 2 seed (0,838 / 0,836 a 0,60M). La soglia al 90%
  dipende dalla taglia a cui il modello "scatta" sulle date (§5), quindi è sensibile anche a piccole
  differenze.
- **Lettura:** con un tokenizer neutro il vantaggio dell'interlingua al 50% scompare (0,98×) e al 90%
  diventa uno svantaggio. H0 ne esce rafforzata: nessuna riduzione di parametri attribuibile al
  formato.

### 8.2 Meno esposizioni: E100 (A_exp, 100 passaggi per biografia)

Pari passi, metà degli step di E200 (3906). Finestra di taglie spostata di uno verso l'alto
(0,20M–0,99M), 3 formati, 2 seed.

| Taglia | italiano_pieno | telegrafico | interlingua |
|---|---|---|---|
| 0,20M | 0,093 | 0,127 | 0,131 |
| 0,31M | 0,173 | 0,268 | 0,282 |
| 0,38M | 0,185 | 0,294 | 0,345 |
| 0,60M | 0,395 | 0,532 | 0,459 |
| 0,79M | 0,454 | 0,662 | 0,547 |
| 0,99M | 0,660 | **0,786** | 0,644 |

| Formato | 50% non-emb | rapporto vs pieno (s0 / s1) | E200 per confronto | tempo medio per run |
|---|---|---|---|---|
| italiano_pieno | 836k | 1 | 1 (244k) | 85 s |
| telegrafico | 573k | **0,69** (0,68 / 0,70) | 0,89 | 67 s |
| interlingua | 689k | **0,82** (0,82 / 0,83) | 0,91 | 74 s |

- **Il 90% non viene mai raggiunto.** Con 100 esposizioni nessun modello impara le date (≤4% a
  tutte le taglie), quindi l'accuratezza resta sotto 5/6 ≈ 0,83.
- **Servono molti più parametri:** con metà delle esposizioni il pieno arriva al 50% solo con 3,4×
  i parametri di E200 (836k contro 244k).
- **Qui la compressione pesa di più:** il telegrafico richiede 0,69× i parametri del pieno, contro
  0,89× a E200, ed è stabile tra i seed. Quando il training è corto, il formato con meno token per
  fatto raggiunge prima la memorizzazione: un effetto di **efficienza del training**, non di capacità.
- **Professione:** è l'attributo più penalizzato nel pieno (0,28 a 0,99M, contro 0,97 nel
  telegrafico), perché il pieno usa più template e forme declinate per lo stesso valore.

### 8.3 Più esposizioni: E500 (A_exp, 500 passaggi per biografia)

Pari passi, 2,5× gli step di E200 (19 531). Taglie 0,15M–0,79M (la finestra spostata in basso
coincide con quella base, perché 0,15M è la taglia minima), 3 formati, 2 seed.

| Taglia | italiano_pieno | telegrafico | interlingua |
|---|---|---|---|
| 0,15M | 0,497 | 0,590 | 0,560 |
| 0,20M | 0,621 | 0,699 | 0,677 |
| 0,31M | 0,779 | 0,865 | 0,801 |
| 0,38M | 0,912 | 0,929 | 0,871 |
| 0,60M | 0,999 | 1,000 | 1,000 |
| 0,79M | 1,000 | 1,000 | 1,000 |

| Formato | 90% non-emb | rapporto vs pieno (s0 / s1) | bit/param max (non-emb) | tempo medio per run |
|---|---|---|---|---|
| italiano_pieno | 381k | 1 | 2,97 | 393 s |
| telegrafico | 352k | **0,92** (0,88 / 1,11) | 3,48 | 300 s |
| interlingua | 430k | **1,13** (1,20 / 1,02) | 3,36 | 336 s |

- **Soglia al 50%:** già superata (o quasi) alla taglia minima da tutti i formati. Il rapporto non è
  misurabile: ≤150k per tutti.
- **Soglia al 90%:** rapporti vicini a 1 e **instabili tra seed**. Il telegrafico va da 0,88 a 1,11 e
  l'interlingua da 1,20 a 1,02, quindi nessun formato ha un vantaggio affidabile.
- **Più esposizioni aiutano tutti:** il pieno raggiunge il 90% con 381k parametri, contro 559k a E200.
- **Bit per parametro:** a 0,15M si arriva a 3,0–3,5 bit per parametro non-embedding, sopra i ~2 di
  letteratura. Va letto con cautela: a 0,15M l'embedding (~250k parametri) è più grande del
  transformer, e sui parametri totali il valore scende a ~1,3 bit/parametro.

### Sintesi dello sweep di esposizioni (rapporto telegrafico / pieno, parametri non-emb)

| Esposizioni | 50% | 90% |
|---|---|---|
| E100 | **0,69** | non raggiunto |
| E200 | 0,89 | 0,90 |
| E500 | ≤150k per tutti, non misurabile | 0,92 (0,88–1,11 tra seed) |

**Il vantaggio del formato compresso si riduce all'aumentare delle esposizioni.** Con poco training
conta l'efficienza (meno token per fatto). Con training abbondante la capacità richiesta converge allo
stesso valore per tutti i formati, entro il rumore tra seed. È il pattern previsto da H0: la forma
cambia la velocità di apprendimento, non i parametri necessari a regime. Grafico:
`results/v2/bits_per_param_vs_exposures.png`.

### 8.4 Pari esposizioni (A_esp)

Ogni biografia è vista 200 volte, ma con **gli stessi token per step** del pieno (~26k). I formati
compressi fanno quindi meno step e meno compute: telegrafico 5277 step (0,68×), interlingua 5865
(0,75×), ricco 12 945 (1,66×). Il riferimento pieno è quello di A_main (identico in entrambi i
protocolli). Taglie 0,15M–0,79M, 2 seed.

| Taglia | italiano_pieno | telegrafico | interlingua | italiano_ricco |
|---|---|---|---|---|
| 0,15M | 0,266 | 0,363 | 0,342 | 0,262 |
| 0,20M | 0,434 | 0,424 | 0,440 | 0,311 |
| 0,31M | 0,581 | 0,720 | 0,632 | 0,550 |
| 0,38M | 0,674 | 0,748 | 0,730 | 0,613 |
| 0,60M | 0,953 | 0,921 | 0,909 | 0,888 |
| 0,79M | 0,997 | 0,999 | 0,988 | 0,961 |

| Formato | rapporto 50% (s0 / s1) | rapporto 90% (s0 / s1) | v1 pari esposizioni 50% / 90% | tempo medio per run |
|---|---|---|---|---|
| telegrafico | **0,92** (0,94 / 0,89) | **1,03** (1,00 / 1,05) | 1,18 / 1,45 | 109 s |
| interlingua | **0,94** (0,98 / 0,89) | **1,06** (0,99 / 1,14) | 1,58 / 2,01 | 124 s |
| italiano_ricco | **1,16** (1,19 / 1,14) | **1,14** (1,11 / 1,15) | – | 271 s |

- **Il forte svantaggio della v1 sparisce.** A pari esposizioni la v1 chiedeva 1,45× (telegrafico) e
  2,0× (interlingua) i parametri al 90%. In v2 i rapporti sono ~1,0–1,06, pur con il 25–32% di step in
  meno. Lo svantaggio della v1 dipendeva dal lr non ottimizzato e dal tokenizer, non dal formato.
- **I formati compressi raggiungono la stessa capacità con ~0,7× il compute.** È il dato più utile in
  pratica: a pari risultato, meno token da elaborare.
- **Il ricco richiede ~1,15× i parametri** anche con 1,66× gli step: il testo ridondante occupa
  capacità, nella stessa misura del protocollo a pari passi (1,13 / 1,09).
- I due protocolli danno rapporti entro ±0,1 l'uno dall'altro: **H0 regge sotto entrambi**.

## 9. Conclusioni H1 / H0

- **Monolingua:** H0 supportata. Riduzioni di parametri ≤ ~11% (telegrafico), nulle o incerte per
  l'interlingua al 90%, aumento ~10% per il testo arricchito, contro variazioni di lunghezza del
  30–65%. La forma linguistica conta al margine; la capacità è dominata dai fatti da memorizzare.
  I blocchi extra lo confermano:
  - con tokenizer condiviso il vantaggio dell'interlingua sparisce (0,98× / 1,28×);
  - a pari esposizioni i rapporti sono ~1,0–1,06 (contro 1,45–2,0 in v1);
  - lungo lo sweep di esposizioni il vantaggio del telegrafico scende da 0,69× (E100) a ~0,9× (E200) e
    a ~1 entro il rumore (E500).

  L'unico effetto robusto della compressione è sul **compute**: meno token per arrivare allo stesso
  risultato, quindi apprendimento più rapido quando il training è scarso.
- **Multilingua:** H0 supportata anche qui. Due lingue nello stesso modello non cambiano i parametri
  necessari (0,94–0,96×, entro il rumore), e la lingua non viene "astratta via": un fatto appreso in
  una lingua non è recuperabile nell'altra se la forma del valore cambia (1–3% di recall a 4M), anche
  con il 10% di esempi ponte. A queste taglie il modello memorizza associazioni tra stringhe, non fatti
  indipendenti dalla lingua; un formato interlingua condiviso darebbe il trasferimento che il modello
  non costruisce da solo, ma non ridurrebbe i parametri per fatto.
- **Costi:** la compressione conviene in due casi:
  - compute di training: 0,62–0,77× i FLOP per arrivare al 90%;
  - generazione di testo lungo: 0,6–0,8× il tempo, stimato.

  Non conviene nella risposta fattuale breve: lì la pipeline con traduzione è 3–7× più lenta.

## 10. Limiti e deviazioni

- **Piano ridotto scelto dall'utente** (budget ~5h per la Fase A): sweep lr su 3 taglie invece di
  tutte; 2 seed nel main (4M con un solo seed); blocchi extra su 6 taglie attorno alla transizione
  e 2 seed; sweep esposizioni senza italiano_ricco.
- **lr al bordo della griglia:** a 0,20–0,38M il 2× è il migliore per tutti i formati; l'ottimo
  potrebbe essere più alto. Essendo uniforme tra formati, i rapporti restano confrontabili, ma i
  valori assoluti delle soglie possono essere leggermente sovrastimati.
- **lr per le taglie non provate:** quello della taglia provata più vicina in log-parametri.
- **Fase B:** domanda cross-lingua sempre con il prompt pieno dell'altra lingua; i valori sono
  confrontati nella forma dell'altra lingua (strict) o in una delle due (lenient). Persone ponte
  escluse dalla valutazione cross. A pari passi, nel multi_bridge ogni biografia è vista ~9% meno volte
  (22 000 biografie per epoca invece di 20 000).
- **Fase C:**
  - la lunghezza della risposta del core compresso è fissata dalla verità (oracolo k);
  - i target di stripper e traduttore sono template deterministici e il traduttore riceve il valore:
    l'exact match 1,000 è quindi un limite superiore;
  - lo scenario di ragionamento è una stima analitica, non una generazione reale, perché i core non
    hanno abbastanza contesto;
  - i tempi sono dominati dall'overhead per passo. Con modelli grandi, limitati dal compute, il
    rapporto di tempo seguirebbe quello dei FLOP, e stripper e traduttore piccoli peserebbero
    relativamente meno;
  - misure su un solo seed per taglia.
- **Valutazione su 5000 persone** (30 000 domande) per run, non su tutte le 20 000.
- **Architettura:** head_dim 16 (necessario per flash attention), profondità mista (3–5 layer);
  la curva ha un gradino tra 0,31M (4 layer, d=80) e 0,38M (5 layer, d=80).
- **Loss in bf16** calcolata solo sulle posizioni non-padding; batching per biografia con padding
  dinamico.
- **Interlingua:** mantiene il marcatore "di" della v1; i valori sono confrontati in forma normalizzata.
- **italiano_ricco:** le professioni non sono declinate al femminile.
- **Vocabolario:** BPE a 4096 ma alcuni corpora saturano prima (3781–3827 token effettivi); i
  parametri totali includono l'embedding effettivo.
