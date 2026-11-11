# Riepilogo esperimento v4

## Stato

| formato | run | taglie | seed |
|---|---|---|---|
| italiano_pieno | 24 | 0.15M, 0.20M, 0.31M, 0.38M, 0.45M, 0.52M, 0.60M, 0.79M | 0, 1, 2 |
| italiano_canonico | 24 | 0.15M, 0.20M, 0.31M, 0.38M, 0.45M, 0.52M, 0.60M, 0.79M | 0, 1, 2 |
| telegrafico_canonico | 24 | 0.15M, 0.20M, 0.31M, 0.38M, 0.45M, 0.52M, 0.60M, 0.79M | 0, 1, 2 |
| telegrafico_lista | 24 | 0.15M, 0.20M, 0.31M, 0.38M, 0.45M, 0.52M, 0.60M, 0.79M | 0, 1, 2 |

## Forme

| formato | tokens_per_bio | bit/bio | attributo | n_forms | n_forms_core |
|---|---|---|---|---|---|
| italiano_pieno | 57.6 | 17.00 | nascita_data | 18 | 8 |
| italiano_pieno | 57.6 | 17.00 | nascita_luogo | 20 | 8 |
| italiano_pieno | 57.6 | 17.00 | universita | 16 | 8 |
| italiano_pieno | 57.6 | 17.00 | corso | 18 | 8 |
| italiano_pieno | 57.6 | 17.00 | professione | 16 | 8 |
| italiano_pieno | 57.6 | 17.00 | azienda | 18 | 8 |
| italiano_canonico | 55.8 | 0.00 | nascita_data | 2 | 1 |
| italiano_canonico | 55.8 | 0.00 | nascita_luogo | 4 | 1 |
| italiano_canonico | 55.8 | 0.00 | universita | 2 | 1 |
| italiano_canonico | 55.8 | 0.00 | corso | 4 | 1 |
| italiano_canonico | 55.8 | 0.00 | professione | 2 | 1 |
| italiano_canonico | 55.8 | 0.00 | azienda | 2 | 1 |
| telegrafico_canonico | 41.5 | 0.00 | nascita_data | 2 | 1 |
| telegrafico_canonico | 41.5 | 0.00 | nascita_luogo | 4 | 1 |
| telegrafico_canonico | 41.5 | 0.00 | universita | 2 | 1 |
| telegrafico_canonico | 41.5 | 0.00 | corso | 4 | 1 |
| telegrafico_canonico | 41.5 | 0.00 | professione | 2 | 1 |
| telegrafico_canonico | 41.5 | 0.00 | azienda | 2 | 1 |
| telegrafico_lista | 40.5 | 13.04 | nascita_data | 10 | 4 |
| telegrafico_lista | 40.5 | 13.04 | nascita_luogo | 12 | 4 |
| telegrafico_lista | 40.5 | 13.04 | universita | 8 | 4 |
| telegrafico_lista | 40.5 | 13.04 | corso | 12 | 5 |
| telegrafico_lista | 40.5 | 13.04 | professione | 10 | 6 |
| telegrafico_lista | 40.5 | 13.04 | azienda | 12 | 6 |

## Soglie

| formato | livello | params_total | [lo, hi] | seed min-max | rapporto vs italiano_pieno |
|---|---|---|---|---|---|
| italiano_canonico | 50% | 413k | [407k, 419k] | 410k-418k | 0.91 |
| italiano_canonico | 90% | 768k | [737k, 782k] | 737k-780k | 1.01 |
| italiano_pieno | 50% | 455k | [449k, 468k] | 448k-468k | 1.00 |
| italiano_pieno | 90% | 760k | [741k, 773k] | 740k-771k | 1.00 |
| telegrafico_canonico | 50% | 389k | [384k, 394k] | 383k-394k | 0.85 |
| telegrafico_canonico | 90% | 726k | [705k, 772k] | 703k-771k | 0.95 |
| telegrafico_lista | 50% | 480k | [458k, 497k] | 456k-496k | 1.05 |
| telegrafico_lista | 90% | 742k | [724k, 787k] | 724k-774k | 0.98 |

## 2x2 per livello

| livello | reg_full | reg_strip | strip_vario | strip_canon | interaction | total | share_reg | share_strip |
|---|---|---|---|---|---|---|---|---|
| 50% | 0.91 [0.88, 0.93] | 0.81 [0.78, 0.85] | 1.05 [1.01, 1.10] | 0.94 [0.92, 0.96] | 0.89 [0.85, 0.94] | 0.85 [0.83, 0.87] | 97.4% [81.8%, 112.1%] | 2.6% [-12.1%, 18.2%] |
| 90% | 1.01 [0.97, 1.04] | 0.98 [0.91, 1.04] | 0.98 [0.95, 1.04] | 0.95 [0.91, 1.01] | 0.97 [0.89, 1.03] | 0.95 [0.92, 1.01] | 12.8% [-377.2%, 305.2%] | 87.2% [-205.2%, 477.2%] |

## Tempo di training al 90%

| formato | params_total 90% | train_time_s 90% |
|---|---|---|
| italiano_canonico | 768k | 118 |
| italiano_pieno | 760k | 137 |
| telegrafico_canonico | 726k | 76 |
| telegrafico_lista | 742k | 90 |

## Soglie per attributo

| formato | attributo | livello | params_total | ratio_reg (canonico / vario) |
|---|---|---|---|---|
| italiano_canonico | nascita_data | 50% | 771k | 1.03 |
| italiano_canonico | nascita_data | 90% | 908k | 1.09 |
| italiano_pieno | nascita_data | 50% | 747k | 1.00 |
| italiano_pieno | nascita_data | 90% | 833k | 1.00 |
| telegrafico_canonico | nascita_data | 50% | 726k | 1.08 |
| telegrafico_canonico | nascita_data | 90% | 834k | 1.03 |
| telegrafico_lista | nascita_data | 50% | 675k | 1.00 |
| telegrafico_lista | nascita_data | 90% | 806k | 1.00 |
| italiano_canonico | nascita_luogo | 50% | 499k | 1.06 |
| italiano_canonico | nascita_luogo | 90% | 586k | 0.82 |
| italiano_pieno | nascita_luogo | 50% | 472k | 1.00 |
| italiano_pieno | nascita_luogo | 90% | 715k | 1.00 |
| telegrafico_canonico | nascita_luogo | 50% | 480k | 0.70 |
| telegrafico_canonico | nascita_luogo | 90% | 647k | 0.66 |
| telegrafico_lista | nascita_luogo | 50% | 683k | 1.00 |
| telegrafico_lista | nascita_luogo | 90% | 985k | 1.00 |
| italiano_canonico | universita | 50% | 418k | 0.99 |
| italiano_canonico | universita | 90% | 607k | 0.93 |
| italiano_pieno | universita | 50% | 423k | 1.00 |
| italiano_pieno | universita | 90% | 651k | 1.00 |
| telegrafico_canonico | universita | 50% | 401k | 1.35 |
| telegrafico_canonico | universita | 90% | 617k | 1.05 |
| telegrafico_lista | universita | 50% | 297k | 1.00 |
| telegrafico_lista | universita | 90% | 585k | 1.00 |
| italiano_canonico | corso | 50% | 309k | 0.89 |
| italiano_canonico | corso | 90% | 451k | 0.93 |
| italiano_pieno | corso | 50% | 349k | 1.00 |
| italiano_pieno | corso | 90% | 484k | 1.00 |
| telegrafico_canonico | corso | 50% | 288k | 1.00 |
| telegrafico_canonico | corso | 90% | 452k | 1.00 |
| telegrafico_lista | corso | 50% | 290k | 1.00 |
| telegrafico_lista | corso | 90% | 452k | 1.00 |
| italiano_canonico | professione | 50% | 371k | 0.71 |
| italiano_canonico | professione | 90% | 502k | 0.70 |
| italiano_pieno | professione | 50% | 523k | 1.00 |
| italiano_pieno | professione | 90% | 713k | 1.00 |
| telegrafico_canonico | professione | 50% | 285k | 0.53 |
| telegrafico_canonico | professione | 90% | 355k | 0.53 |
| telegrafico_lista | professione | 50% | 533k | 1.00 |
| telegrafico_lista | professione | 90% | 673k | 1.00 |
| italiano_canonico | azienda | 50% | 320k | 0.93 |
| italiano_canonico | azienda | 90% | 485k | 1.02 |
| italiano_pieno | azienda | 50% | 343k | 1.00 |
| italiano_pieno | azienda | 90% | 478k | 1.00 |
| telegrafico_canonico | azienda | 50% | 368k | 0.87 |
| telegrafico_canonico | azienda | 90% | 506k | 0.82 |
| telegrafico_lista | azienda | 50% | 421k | 1.00 |
| telegrafico_lista | azienda | 90% | 615k | 1.00 |

## Accuratezza per attributo (0.31M-0.52M, media sui seed)

| formato | taglia | params_total | acc_nascita_data | acc_nascita_luogo | acc_universita | acc_corso | acc_professione | acc_azienda | acc_mean |
|---|---|---|---|---|---|---|---|---|---|
| italiano_pieno | 0.31M | 482k | 0.021 | 0.520 | 0.604 | 0.898 | 0.321 | 0.910 | 0.546 |
| italiano_pieno | 0.38M | 560k | 0.021 | 0.724 | 0.758 | 0.944 | 0.646 | 0.945 | 0.673 |
| italiano_pieno | 0.45M | 652k | 0.214 | 0.795 | 0.901 | 0.971 | 0.780 | 0.977 | 0.773 |
| italiano_pieno | 0.52M | 746k | 0.498 | 0.947 | 0.956 | 0.973 | 0.960 | 0.985 | 0.886 |
| italiano_canonico | 0.31M | 481k | 0.004 | 0.381 | 0.701 | 0.948 | 0.868 | 0.896 | 0.633 |
| italiano_canonico | 0.38M | 559k | 0.008 | 0.870 | 0.834 | 0.959 | 0.979 | 0.961 | 0.769 |
| italiano_canonico | 0.45M | 651k | 0.047 | 0.967 | 0.955 | 0.985 | 0.991 | 0.982 | 0.821 |
| italiano_canonico | 0.52M | 745k | 0.365 | 0.976 | 0.969 | 0.984 | 0.991 | 0.986 | 0.878 |
| telegrafico_canonico | 0.31M | 480k | 0.005 | 0.499 | 0.755 | 0.947 | 0.978 | 0.878 | 0.677 |
| telegrafico_canonico | 0.38M | 558k | 0.019 | 0.622 | 0.845 | 0.963 | 0.983 | 0.942 | 0.729 |
| telegrafico_canonico | 0.45M | 650k | 0.085 | 0.909 | 0.928 | 0.986 | 0.993 | 0.977 | 0.813 |
| telegrafico_canonico | 0.52M | 744k | 0.593 | 0.976 | 0.976 | 0.990 | 0.996 | 0.983 | 0.919 |
| telegrafico_lista | 0.31M | 480k | 0.082 | 0.132 | 0.845 | 0.952 | 0.365 | 0.627 | 0.500 |
| telegrafico_lista | 0.38M | 558k | 0.110 | 0.263 | 0.889 | 0.962 | 0.561 | 0.784 | 0.595 |
| telegrafico_lista | 0.45M | 650k | 0.380 | 0.377 | 0.924 | 0.983 | 0.878 | 0.965 | 0.751 |
| telegrafico_lista | 0.52M | 744k | 0.813 | 0.717 | 0.963 | 0.986 | 0.963 | 0.976 | 0.903 |

## Verifica lr

| formato | taglia | lr_mult | acc_mean | final_loss |
|---|---|---|---|---|
| italiano_canonico | 0.38M | 1 | 0.6279 | 0.5328 |
| italiano_canonico | 0.38M | 2 | 0.739 | 0.4713 |
| italiano_canonico | 0.38M | 4 | 0.6853 | 0.5048 |

## Allineamento: align_bench

| vocab | V | aligned | V_mod64 | T | step_ms | head_ms |
|---|---|---|---|---|---|---|
| bpe4096 | 3814 | False | 38 | 64 | 11.1 | 4.169 |
| bpe4096 | 3816 | False | 40 | 64 | 11.02 | 3.246 |
| bpe4096 | 3840 | True | 0 | 64 | 11.06 | 3.237 |
| bpe2048 | 2047 | False | 63 | 78 | 13.08 | 1.904 |
| bpe2048 | 2040 | False | 56 | 78 | 12.37 | 1.852 |
| bpe2048 | 2048 | True | 0 | 78 | 12.9 | 1.875 |

## Allineamento: align_control

| corpus | riferimento | V | V_padded | acc_ref | acc_aligned | delta | train_time_ref_s | train_time_aligned_s | params_total |
|---|---|---|---|---|---|---|---|---|---|
| italiano_pieno@bpe2048 | v3 D_main | 2048 | 2048 | 0.6755 | 0.663 | -0.0125 | 104.8 | 104.9 | 559600 |
| italiano_pieno | v2 A_main | 3814 | 3840 | 0.6733 | 0.646 | -0.0273 | 169.9 | 108.4 | 699840 |
