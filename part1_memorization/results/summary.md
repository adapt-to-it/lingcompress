# Riepilogo esperimento

## Token per biografia (media per formato)

| format | tokens_per_bio |
|---|---|
| italiano_pieno | 45.08 |
| telegrafico | 28.58 |
| interlingua | 24.58 |

## Soglie di accuratezza

| format | level | params | lower_bound | ratio_vs_pieno |
|---|---|---|---|---|
| italiano_pieno | 0.5 | 4e+05 | False | 1 |
| italiano_pieno | 0.9 | 8.569e+05 | False | 1 |
| telegrafico | 0.5 | 4.731e+05 | False | 1.183 |
| telegrafico | 0.9 | 1.241e+06 | False | 1.448 |
| interlingua | 0.5 | 6.324e+05 | False | 1.581 |
| interlingua | 0.9 | 1.722e+06 | False | 2.01 |

## Totali per formato (somma su tutte le run)

| format | train_tokens | flops | train_time_s | train_tokens_vs_pieno | flops_vs_pieno | train_time_s_vs_pieno |
|---|---|---|---|---|---|---|
| italiano_pieno | 2235715644 | 44516711388520704 | 7452 | 1 | 1 | 1 |
| telegrafico | 1432029564 | 28468656082427904 | 3771 | 0.6405 | 0.6395 | 0.5061 |
| interlingua | 1238845044 | 24663838925904384 | 3337 | 0.5541 | 0.554 | 0.4478 |

## Accuratezza per formato e taglia, con attributi

| format | size | params_nonemb | acc_mean | acc_nascita_data | acc_nascita_luogo | acc_universita | acc_corso | acc_professione | acc_azienda |
|---|---|---|---|---|---|---|---|---|---|
| italiano_pieno | 0.25M | 2.527e+05 | 0.208 | 0.056 | 0.3194 | 0.1447 | 0.2211 | 0.08322 | 0.4237 |
| italiano_pieno | 0.5M | 5.248e+05 | 0.6727 | 0.4253 | 0.9374 | 0.754 | 0.8302 | 0.1998 | 0.8895 |
| italiano_pieno | 1M | 1.046e+06 | 0.9926 | 0.9964 | 0.9997 | 0.9973 | 0.9988 | 0.9638 | 0.9997 |
| italiano_pieno | 2M | 2.046e+06 | 1 | 0.9999 | 1 | 1 | 1 | 0.9999 | 1 |
| italiano_pieno | 4M | 4.175e+06 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| italiano_pieno | 8M | 7.993e+06 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| telegrafico | 0.25M | 2.527e+05 | 0.1092 | 0.00225 | 0.1727 | 0.07212 | 0.06105 | 0.1529 | 0.1944 |
| telegrafico | 0.5M | 5.248e+05 | 0.5646 | 0.08528 | 0.8341 | 0.4915 | 0.4124 | 0.7296 | 0.8348 |
| telegrafico | 1M | 1.046e+06 | 0.8659 | 0.6021 | 0.9739 | 0.83 | 0.8563 | 0.9591 | 0.9741 |
| telegrafico | 2M | 2.046e+06 | 0.9998 | 0.9997 | 0.9999 | 0.9996 | 0.9998 | 1 | 1 |
| telegrafico | 4M | 4.175e+06 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| telegrafico | 8M | 7.993e+06 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| interlingua | 0.25M | 2.527e+05 | 0.04881 | 0.001075 | 0.07265 | 0.029 | 0.034 | 0.0749 | 0.08122 |
| interlingua | 0.5M | 5.248e+05 | 0.4553 | 0.05615 | 0.7151 | 0.1951 | 0.4349 | 0.6219 | 0.7088 |
| interlingua | 1M | 1.046e+06 | 0.6206 | 0.2062 | 0.864 | 0.4707 | 0.5753 | 0.7362 | 0.871 |
| interlingua | 2M | 2.046e+06 | 0.9965 | 0.9992 | 0.9999 | 0.9808 | 0.9994 | 1 | 0.9994 |
| interlingua | 4M | 4.175e+06 | 0.9998 | 1 | 1 | 0.9993 | 0.9998 | 1 | 0.9999 |
| interlingua | 8M | 7.993e+06 | 1 | 1 | 1 | 0.9999 | 0.9999 | 1 | 1 |
