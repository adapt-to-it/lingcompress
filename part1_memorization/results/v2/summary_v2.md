# Riepilogo esperimento v2

## Learning rate scelto per corpus e taglia

| corpus | size | lr_mult scelto | acc_mean | final_loss |
|---|---|---|---|---|
| interlingua | 0.20M | 2 | 0.4119 | 0.947 |
| interlingua | 0.38M | 2 | 0.7128 | 0.7536 |
| interlingua | 0.79M | 1 | 0.9948 | 0.4393 |
| italiano_pieno | 0.20M | 2 | 0.4433 | 0.9424 |
| italiano_pieno | 0.38M | 2 | 0.6733 | 0.7985 |
| italiano_pieno | 0.79M | 1 | 0.9972 | 0.5674 |
| italiano_ricco | 0.20M | 2 | 0.2084 | 0.8223 |
| italiano_ricco | 0.38M | 2 | 0.5444 | 0.6991 |
| italiano_ricco | 0.79M | 2 | 0.9729 | 0.5467 |
| telegrafico | 0.20M | 2 | 0.4154 | 1.175 |
| telegrafico | 0.38M | 2 | 0.8009 | 0.8491 |
| telegrafico | 0.79M | 1 | 0.9998 | 0.6007 |

## Soglie di accuratezza, A_main (rapporti vs italiano_pieno)

| corpus | livello | params_nonemb | params_total | lower_bound | n_seeds | rapporto_nonemb_vs_italiano_pieno | rapporto_total_vs_italiano_pieno |
|---|---|---|---|---|---|---|---|
| interlingua | 50% | 2.231e+05 | 4.86e+05 | False | 2 | 0.9132 | 0.935 |
| interlingua | 90% | 5.725e+05 | 9.879e+05 | False | 2 | 1.024 | 1.021 |
| italiano_pieno | 50% | 2.443e+05 | 5.198e+05 | False | 2 | 1 | 1 |
| italiano_pieno | 90% | 5.59e+05 | 9.672e+05 | False | 2 | 1 | 1 |
| italiano_ricco | 50% | 2.768e+05 | 5.965e+05 | False | 2 | 1.133 | 1.148 |
| italiano_ricco | 90% | 6.064e+05 | 1.079e+06 | False | 2 | 1.085 | 1.116 |
| telegrafico | 50% | 2.185e+05 | 4.749e+05 | False | 2 | 0.8943 | 0.9136 |
| telegrafico | 90% | 5.015e+05 | 8.722e+05 | False | 2 | 0.8971 | 0.9018 |

## Soglie per numero di esposizioni, A_esp (rapporti vs italiano_pieno allo stesso numero)

| corpus | exposures | livello | params_nonemb | params_total | lower_bound | n_seeds | rapporto_nonemb_vs_italiano_pieno | rapporto_total_vs_italiano_pieno |
|---|---|---|---|---|---|---|---|---|
| interlingua | 200 | 50% | 2.298e+05 | 4.967e+05 | False | 2 | 0.9405 | 0.9556 |
| interlingua | 200 | 90% | 5.946e+05 | 1.022e+06 | False | 2 | 1.064 | 1.057 |
| italiano_pieno | 200 | 50% | 2.443e+05 | 5.198e+05 | False | 2 | 1 | 1 |
| italiano_pieno | 200 | 90% | 5.59e+05 | 9.672e+05 | False | 2 | 1 | 1 |
| italiano_ricco | 200 | 50% | 2.838e+05 | 6.075e+05 | False | 2 | 1.162 | 1.169 |
| italiano_ricco | 200 | 90% | 6.361e+05 | 1.121e+06 | False | 2 | 1.138 | 1.159 |
| telegrafico | 200 | 50% | 2.242e+05 | 4.84e+05 | False | 2 | 0.9174 | 0.931 |
| telegrafico | 200 | 90% | 5.758e+05 | 9.871e+05 | False | 2 | 1.03 | 1.021 |

## Soglie con tokenizer condiviso, A_shared (rapporti vs italiano_pieno)

| corpus | livello | params_nonemb | params_total | lower_bound | n_seeds | rapporto_nonemb_vs_italiano_pieno | rapporto_total_vs_italiano_pieno |
|---|---|---|---|---|---|---|---|
| interlingua@shared | 50% | 2.197e+05 | 4.985e+05 | False | 2 | 0.899 | 0.9589 |
| interlingua@shared | 90% | 7.008e+05 | 1.2e+06 | False | 2 | 1.254 | 1.241 |
| italiano_pieno | 50% | 2.443e+05 | 5.198e+05 | False | 2 | 1 | 1 |
| italiano_pieno | 90% | 5.59e+05 | 9.672e+05 | False | 2 | 1 | 1 |
| italiano_pieno@shared | 50% | 2.245e+05 | 5.076e+05 | False | 2 | 0.9188 | 0.9764 |
| italiano_pieno@shared | 90% | 5.471e+05 | 9.78e+05 | False | 2 | 0.9786 | 1.011 |

## Ablazioni, A_abl (rapporti vs corpus base)

| corpus | livello | params_nonemb | params_total | lower_bound | n_seeds | rapporto_nonemb_vs_base | rapporto_total_vs_base |
|---|---|---|---|---|---|---|---|
| interlingua | 50% | 2.231e+05 | 4.86e+05 | False | 2 | 1 | 1 |
| interlingua | 90% | 5.725e+05 | 9.879e+05 | False | 2 | 1 | 1 |
| interlingua_v1enc | 50% | 2.054e+05 | 4.554e+05 | False | 2 | 0.9207 | 0.9369 |
| interlingua_v1enc | 90% | 4.956e+05 | 8.656e+05 | False | 2 | 0.8657 | 0.8762 |
| italiano_pieno | 50% | 2.443e+05 | 5.198e+05 | False | 2 | 1 | 1 |
| italiano_pieno | 90% | 5.59e+05 | 9.672e+05 | False | 2 | 1 | 1 |
| italiano_pieno@nodigits | 50% | 2.852e+05 | 5.895e+05 | False | 2 | 1.167 | 1.134 |
| italiano_pieno@nodigits | 90% | 5.519e+05 | 9.653e+05 | False | 2 | 0.9872 | 0.9981 |

## Soglie multilingua, B_main (rapporti vs interlingua e vs italiano_pieno)

| corpus | livello | params_nonemb | params_total | lower_bound | n_seeds | rapporto_nonemb_vs_interlingua | rapporto_total_vs_interlingua | rapporto_nonemb_vs_italiano_pieno | rapporto_total_vs_italiano_pieno |
|---|---|---|---|---|---|---|---|---|---|
| interlingua | 50% | 2.231e+05 | 4.86e+05 | False | 2 | 1 | 1 | 0.9132 | 0.935 |
| interlingua | 90% | 5.725e+05 | 9.879e+05 | False | 2 | 1 | 1 | 1.024 | 1.021 |
| italiano_pieno | 50% | 2.443e+05 | 5.198e+05 | False | 2 | 1.095 | 1.07 | 1 | 1 |
| italiano_pieno | 90% | 5.59e+05 | 9.672e+05 | False | 2 | 0.9765 | 0.979 | 1 | 1 |
| multi_bridge | 50% | 2.344e+05 | 5.241e+05 | False | 3 | 1.051 | 1.078 | 0.9595 | 1.008 |
| multi_bridge | 90% | 5.33e+05 | 9.559e+05 | False | 3 | 0.931 | 0.9676 | 0.9534 | 0.9883 |
| multi_pieno | 50% | 2.29e+05 | 5.152e+05 | False | 3 | 1.026 | 1.06 | 0.9372 | 0.991 |
| multi_pieno | 90% | 5.359e+05 | 9.605e+05 | False | 3 | 0.936 | 0.9723 | 0.9586 | 0.9931 |

## Bit per parametro per corpus ed esposizioni (massimo sulle taglie)

| corpus | 100 | 200 | 500 |
|---|---|---|---|
| interlingua | 0.7897 | 2.213 | 3.227 |
| italiano_pieno | 0.571 | 1.885 | 2.856 |
| italiano_ricco | NaN | 1.635 | NaN |
| telegrafico | 0.7528 | 2.208 | 3.384 |

## Recall cross-lingua per taglia (B_main)

| corpus | size | params_nonemb | cross_mean | cross_lenient_mean |
|---|---|---|---|---|
| multi_bridge | 0.15M | 1.501e+05 | 0.1133 | 0.1134 |
| multi_bridge | 0.20M | 2.001e+05 | 0.1634 | 0.1634 |
| multi_bridge | 0.31M | 3.115e+05 | 0.2799 | 0.28 |
| multi_bridge | 0.38M | 3.894e+05 | 0.2723 | 0.2725 |
| multi_bridge | 0.60M | 6.082e+05 | 0.3557 | 0.357 |
| multi_bridge | 0.79M | 7.933e+05 | 0.3471 | 0.3484 |
| multi_bridge | 0.99M | 1.003e+06 | 0.3833 | 0.3854 |
| multi_bridge | 1.49M | 1.496e+06 | 0.4109 | 0.4132 |
| multi_bridge | 4M | 3.997e+06 | 0.4649 | 0.4696 |
| multi_pieno | 0.15M | 1.501e+05 | 0.05049 | 0.05054 |
| multi_pieno | 0.20M | 2.001e+05 | 0.05452 | 0.05457 |
| multi_pieno | 0.31M | 3.115e+05 | 0.1096 | 0.1097 |
| multi_pieno | 0.38M | 3.894e+05 | 0.09121 | 0.09131 |
| multi_pieno | 0.60M | 6.082e+05 | 0.1494 | 0.1498 |
| multi_pieno | 0.79M | 7.933e+05 | 0.1574 | 0.158 |
| multi_pieno | 0.99M | 1.003e+06 | 0.1894 | 0.1903 |
| multi_pieno | 1.49M | 1.496e+06 | 0.2065 | 0.208 |
| multi_pieno | 4M | 3.997e+06 | 0.2533 | 0.2573 |

## Velocità e tempo di addestramento per stage e corpus

| stage | corpus | steady_tokens_per_s (media) | steady_facts_per_s (media) | train_time_s (somma) |
|---|---|---|---|---|
| A_abl | interlingua_v1enc | 1.136e+06 | 2.179e+05 | 1351 |
| A_abl | italiano_pieno@nodigits | 1.276e+06 | 1.661e+05 | 1774 |
| A_esp | interlingua | 1.288e+06 | 1.976e+05 | 1490 |
| A_esp | italiano_ricco | 1.29e+06 | 9.101e+04 | 3251 |
| A_esp | telegrafico | 1.326e+06 | 2.254e+05 | 1309 |
| A_exp | interlingua | 1.134e+06 | 1.74e+05 | 4925 |
| A_exp | italiano_pieno | 1.291e+06 | 1.495e+05 | 5736 |
| A_exp | telegrafico | 1.141e+06 | 1.939e+05 | 4406 |
| A_lr | interlingua | 1.124e+06 | 1.725e+05 | 1276 |
| A_lr | italiano_pieno | 1.28e+06 | 1.482e+05 | 1481 |
| A_lr | italiano_ricco | 1.498e+06 | 1.057e+05 | 2085 |
| A_lr | telegrafico | 1.139e+06 | 1.936e+05 | 1134 |
| A_main | interlingua | 1.068e+06 | 1.639e+05 | 2644 |
| A_main | italiano_pieno | 1.23e+06 | 1.425e+05 | 3018 |
| A_main | italiano_ricco | 1.374e+06 | 9.694e+04 | 4683 |
| A_main | telegrafico | 1.067e+06 | 1.814e+05 | 2392 |
| A_shared | interlingua@shared | 1.273e+06 | 1.927e+05 | 1531 |
| A_shared | italiano_pieno@shared | 1.418e+06 | 1.625e+05 | 1814 |
| B_main | multi_bridge | 1.214e+06 | 1.359e+05 | 5166 |
| B_main | multi_pieno | 1.214e+06 | 1.36e+05 | 5169 |

## Confronto con v1 (soglie su params_nonemb)

| fonte | corpus/format | livello | params_nonemb | lower_bound | rapporto_vs_italiano_pieno |
|---|---|---|---|---|---|
| v2 main | interlingua | 50% | 2.231e+05 | False | 0.9132 |
| v2 main | interlingua | 90% | 5.725e+05 | False | 1.024 |
| v2 main | italiano_pieno | 50% | 2.443e+05 | False | 1 |
| v2 main | italiano_pieno | 90% | 5.59e+05 | False | 1 |
| v2 main | italiano_ricco | 50% | 2.768e+05 | False | 1.133 |
| v2 main | italiano_ricco | 90% | 6.064e+05 | False | 1.085 |
| v2 main | telegrafico | 50% | 2.185e+05 | False | 0.8943 |
| v2 main | telegrafico | 90% | 5.015e+05 | False | 0.8971 |
| v1 pari esposizioni | italiano_pieno | 50% | 4e+05 | False | 1 |
| v1 pari esposizioni | italiano_pieno | 90% | 8.569e+05 | False | 1 |
| v1 pari esposizioni | telegrafico | 50% | 4.731e+05 | False | 1.183 |
| v1 pari esposizioni | telegrafico | 90% | 1.241e+06 | False | 1.448 |
| v1 pari esposizioni | interlingua | 50% | 6.324e+05 | False | 1.581 |
| v1 pari esposizioni | interlingua | 90% | 1.722e+06 | False | 2.01 |
| v1 pari passi | italiano_pieno | 50% | 4e+05 | False | 1 |
| v1 pari passi | italiano_pieno | 90% | 8.569e+05 | False | 1 |
| v1 pari passi | telegrafico | 50% | 3.624e+05 | False | 0.906 |
| v1 pari passi | telegrafico | 90% | 7.609e+05 | False | 0.888 |
| v1 pari passi | interlingua | 50% | 4.021e+05 | False | 1.005 |
| v1 pari passi | interlingua | 90% | 8.963e+05 | False | 1.046 |
