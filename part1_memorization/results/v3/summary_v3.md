# Riepilogo esperimento v3

## D: accuratezza per formato, vocabolario e taglia (media sui seed)

| format | vocab | size | params_nonemb | acc_mean |
|---|---|---|---|---|
| interlingua_v1enc | bpe1024 | 0.15M | 1.501e+05 | 0.1181 |
| interlingua_v1enc | bpe1024 | 0.20M | 2.001e+05 | 0.1972 |
| interlingua_v1enc | bpe1024 | 0.31M | 3.115e+05 | 0.4844 |
| interlingua_v1enc | bpe1024 | 0.38M | 3.894e+05 | 0.6092 |
| interlingua_v1enc | bpe1024 | 0.60M | 6.082e+05 | 0.9256 |
| interlingua_v1enc | bpe1024 | 0.79M | 7.933e+05 | 0.9943 |
| interlingua_v1enc | bpe2048 | 0.15M | 1.501e+05 | 0.2678 |
| interlingua_v1enc | bpe2048 | 0.20M | 2.001e+05 | 0.3634 |
| interlingua_v1enc | bpe2048 | 0.31M | 3.115e+05 | 0.6619 |
| interlingua_v1enc | bpe2048 | 0.38M | 3.894e+05 | 0.7462 |
| interlingua_v1enc | bpe2048 | 0.60M | 6.082e+05 | 0.9659 |
| interlingua_v1enc | bpe2048 | 0.79M | 7.933e+05 | 0.9975 |
| interlingua_v1enc | bpe4096 | 0.15M | 1.501e+05 | 0.3824 |
| interlingua_v1enc | bpe4096 | 0.20M | 2.001e+05 | 0.4836 |
| interlingua_v1enc | bpe4096 | 0.31M | 3.115e+05 | 0.7587 |
| interlingua_v1enc | bpe4096 | 0.38M | 3.894e+05 | 0.7941 |
| interlingua_v1enc | bpe4096 | 0.60M | 6.082e+05 | 0.9898 |
| interlingua_v1enc | bpe4096 | 0.79M | 7.933e+05 | 0.9995 |
| interlingua_v1enc | bpe512 | 0.15M | 1.501e+05 | 0.04098 |
| interlingua_v1enc | bpe512 | 0.20M | 2.001e+05 | 0.09198 |
| interlingua_v1enc | bpe512 | 0.31M | 3.115e+05 | 0.3214 |
| interlingua_v1enc | bpe512 | 0.38M | 3.894e+05 | 0.4208 |
| interlingua_v1enc | bpe512 | 0.60M | 6.082e+05 | 0.834 |
| interlingua_v1enc | bpe512 | 0.79M | 7.933e+05 | 0.9668 |
| interlingua_v1enc | char | 0.15M | 1.501e+05 | 0.0124 |
| interlingua_v1enc | char | 0.20M | 2.001e+05 | 0.02443 |
| interlingua_v1enc | char | 0.31M | 3.115e+05 | 0.06147 |
| interlingua_v1enc | char | 0.38M | 3.894e+05 | 0.1806 |
| interlingua_v1enc | char | 0.60M | 6.082e+05 | 0.5001 |
| interlingua_v1enc | char | 0.79M | 7.933e+05 | 0.6148 |
| italiano_pieno | bpe1024 | 0.15M | 1.501e+05 | 0.0798 |
| italiano_pieno | bpe1024 | 0.20M | 2.001e+05 | 0.1393 |
| italiano_pieno | bpe1024 | 0.31M | 3.115e+05 | 0.4113 |
| italiano_pieno | bpe1024 | 0.38M | 3.894e+05 | 0.5113 |
| italiano_pieno | bpe1024 | 0.60M | 6.082e+05 | 0.8648 |
| italiano_pieno | bpe1024 | 0.79M | 7.933e+05 | 0.9836 |
| italiano_pieno | bpe2048 | 0.15M | 1.501e+05 | 0.1853 |
| italiano_pieno | bpe2048 | 0.20M | 2.001e+05 | 0.2501 |
| italiano_pieno | bpe2048 | 0.31M | 3.115e+05 | 0.5429 |
| italiano_pieno | bpe2048 | 0.38M | 3.894e+05 | 0.6864 |
| italiano_pieno | bpe2048 | 0.60M | 6.082e+05 | 0.9711 |
| italiano_pieno | bpe2048 | 0.79M | 7.933e+05 | 0.9964 |
| italiano_pieno | bpe4096 | 0.15M | 1.501e+05 | 0.2662 |
| italiano_pieno | bpe4096 | 0.20M | 2.001e+05 | 0.4336 |
| italiano_pieno | bpe4096 | 0.31M | 3.115e+05 | 0.5807 |
| italiano_pieno | bpe4096 | 0.38M | 3.894e+05 | 0.6744 |
| italiano_pieno | bpe4096 | 0.60M | 6.082e+05 | 0.9525 |
| italiano_pieno | bpe4096 | 0.79M | 7.933e+05 | 0.9969 |
| italiano_pieno | bpe512 | 0.15M | 1.501e+05 | 0.02007 |
| italiano_pieno | bpe512 | 0.20M | 2.001e+05 | 0.06275 |
| italiano_pieno | bpe512 | 0.31M | 3.115e+05 | 0.2039 |
| italiano_pieno | bpe512 | 0.38M | 3.894e+05 | 0.3528 |
| italiano_pieno | bpe512 | 0.60M | 6.082e+05 | 0.6763 |
| italiano_pieno | bpe512 | 0.79M | 7.933e+05 | 0.9032 |
| italiano_pieno | char | 0.15M | 1.501e+05 | 0.01077 |
| italiano_pieno | char | 0.20M | 2.001e+05 | 0.01403 |
| italiano_pieno | char | 0.31M | 3.115e+05 | 0.0439 |
| italiano_pieno | char | 0.38M | 3.894e+05 | 0.1012 |
| italiano_pieno | char | 0.60M | 6.082e+05 | 0.2699 |
| italiano_pieno | char | 0.79M | 7.933e+05 | 0.5053 |
| telegrafico | bpe1024 | 0.15M | 1.501e+05 | 0.09047 |
| telegrafico | bpe1024 | 0.20M | 2.001e+05 | 0.1745 |
| telegrafico | bpe1024 | 0.31M | 3.115e+05 | 0.4108 |
| telegrafico | bpe1024 | 0.38M | 3.894e+05 | 0.5901 |
| telegrafico | bpe1024 | 0.60M | 6.082e+05 | 0.9481 |
| telegrafico | bpe1024 | 0.79M | 7.933e+05 | 0.998 |
| telegrafico | bpe2048 | 0.15M | 1.501e+05 | 0.247 |
| telegrafico | bpe2048 | 0.20M | 2.001e+05 | 0.3392 |
| telegrafico | bpe2048 | 0.31M | 3.115e+05 | 0.6414 |
| telegrafico | bpe2048 | 0.38M | 3.894e+05 | 0.7042 |
| telegrafico | bpe2048 | 0.60M | 6.082e+05 | 0.9672 |
| telegrafico | bpe2048 | 0.79M | 7.933e+05 | 0.9995 |
| telegrafico | bpe4096 | 0.15M | 1.501e+05 | 0.3847 |
| telegrafico | bpe4096 | 0.20M | 2.001e+05 | 0.4324 |
| telegrafico | bpe4096 | 0.31M | 3.115e+05 | 0.7714 |
| telegrafico | bpe4096 | 0.38M | 3.894e+05 | 0.8051 |
| telegrafico | bpe4096 | 0.60M | 6.082e+05 | 0.9723 |
| telegrafico | bpe4096 | 0.79M | 7.933e+05 | 0.9997 |
| telegrafico | bpe512 | 0.15M | 1.501e+05 | 0.0381 |
| telegrafico | bpe512 | 0.20M | 2.001e+05 | 0.08865 |
| telegrafico | bpe512 | 0.31M | 3.115e+05 | 0.2247 |
| telegrafico | bpe512 | 0.38M | 3.894e+05 | 0.3618 |
| telegrafico | bpe512 | 0.60M | 6.082e+05 | 0.8349 |
| telegrafico | bpe512 | 0.79M | 7.933e+05 | 0.99 |
| telegrafico | char | 0.15M | 1.501e+05 | 0.009167 |
| telegrafico | char | 0.20M | 2.001e+05 | 0.02027 |
| telegrafico | char | 0.31M | 3.115e+05 | 0.09253 |
| telegrafico | char | 0.38M | 3.894e+05 | 0.1238 |
| telegrafico | char | 0.60M | 6.082e+05 | 0.327 |
| telegrafico | char | 0.79M | 7.933e+05 | 0.664 |

## D: token per bio e dimensione del vocabolario

| format | vocab | tokens_per_bio | vocab_size |
|---|---|---|---|
| interlingua_v1enc | bpe1024 | 45.52 | 1024 |
| interlingua_v1enc | bpe2048 | 36.93 | 2048 |
| interlingua_v1enc | bpe4096 | 30.29 | 3811 |
| interlingua_v1enc | bpe512 | 57.64 | 512 |
| interlingua_v1enc | char | 137.8 | 70 |
| italiano_pieno | bpe1024 | 66.61 | 1024 |
| italiano_pieno | bpe2048 | 57.55 | 2048 |
| italiano_pieno | bpe4096 | 50.76 | 3814 |
| italiano_pieno | bpe512 | 83.58 | 512 |
| italiano_pieno | char | 230.8 | 72 |
| telegrafico | bpe1024 | 49.57 | 1024 |
| telegrafico | bpe2048 | 40.83 | 2048 |
| telegrafico | bpe4096 | 34.29 | 3781 |
| telegrafico | bpe512 | 63.44 | 512 |
| telegrafico | char | 170.3 | 70 |

## D: soglie 50% e 90% (rapporti vs italiano_pieno allo stesso vocabolario)

| format | vocab | livello | params_nonemb | params_total | lower_bound | n_seeds | rapporto_nonemb_vs_pieno_stesso_vocab | rapporto_total_vs_pieno_stesso_vocab |
|---|---|---|---|---|---|---|---|---|
| interlingua_v1enc | bpe1024 | 50% | 3.203e+05 | 4.08e+05 | False | 2 | 0.8437 | 0.8687 |
| interlingua_v1enc | bpe1024 | 90% | 5.866e+05 | 7.059e+05 | False | 2 | 0.8915 | 0.8946 |
| interlingua_v1enc | bpe2048 | 50% | 2.45e+05 | 3.944e+05 | False | 2 | 0.8393 | 0.863 |
| interlingua_v1enc | bpe2048 | 90% | 5.32e+05 | 7.448e+05 | False | 2 | 0.9778 | 0.9757 |
| interlingua_v1enc | bpe4096 | 50% | 2.054e+05 | 4.554e+05 | False | 2 | 0.8407 | 0.876 |
| interlingua_v1enc | bpe4096 | 90% | 4.956e+05 | 8.656e+05 | False | 2 | 0.8866 | 0.8949 |
| interlingua_v1enc | bpe512 | 50% | 4.241e+05 | 4.754e+05 | False | 2 | 0.8892 | 0.8861 |
| interlingua_v1enc | bpe512 | 90% | 6.941e+05 | 7.66e+05 | False | 2 | 0.8781 | 0.8787 |
| interlingua_v1enc | char | 50% | 6.081e+05 | 6.371e+05 | False | 1 | 0.7711 | 0.7593 |
| interlingua_v1enc | char | 90% | NaN | NaN | False | 1 | NaN | NaN |
| italiano_pieno | bpe1024 | 50% | 3.796e+05 | 4.697e+05 | False | 2 | 1 | 1 |
| italiano_pieno | bpe1024 | 90% | 6.58e+05 | 7.89e+05 | False | 2 | 1 | 1 |
| italiano_pieno | bpe2048 | 50% | 2.919e+05 | 4.57e+05 | False | 2 | 1 | 1 |
| italiano_pieno | bpe2048 | 90% | 5.441e+05 | 7.634e+05 | False | 2 | 1 | 1 |
| italiano_pieno | bpe4096 | 50% | 2.443e+05 | 5.198e+05 | False | 2 | 1 | 1 |
| italiano_pieno | bpe4096 | 90% | 5.59e+05 | 9.672e+05 | False | 2 | 1 | 1 |
| italiano_pieno | bpe512 | 50% | 4.77e+05 | 5.365e+05 | False | 2 | 1 | 1 |
| italiano_pieno | bpe512 | 90% | 7.904e+05 | 8.718e+05 | False | 2 | 1 | 1 |
| italiano_pieno | char | 50% | 7.886e+05 | 8.39e+05 | False | 1 | 1 | 1 |
| italiano_pieno | char | 90% | NaN | NaN | False | 1 | NaN | NaN |
| telegrafico | bpe1024 | 50% | 3.481e+05 | 4.364e+05 | False | 2 | 0.9169 | 0.9291 |
| telegrafico | bpe1024 | 90% | 5.728e+05 | 6.904e+05 | False | 2 | 0.8705 | 0.875 |
| telegrafico | bpe2048 | 50% | 2.532e+05 | 4.054e+05 | False | 2 | 0.8673 | 0.8871 |
| telegrafico | bpe2048 | 90% | 5.427e+05 | 7.591e+05 | False | 2 | 0.9975 | 0.9944 |
| telegrafico | bpe4096 | 50% | 2.185e+05 | 4.749e+05 | False | 2 | 0.8943 | 0.9136 |
| telegrafico | bpe4096 | 90% | 5.015e+05 | 8.722e+05 | False | 2 | 0.8971 | 0.9018 |
| telegrafico | bpe512 | 50% | 4.435e+05 | 4.975e+05 | False | 2 | 0.9299 | 0.9274 |
| telegrafico | bpe512 | 90% | 6.799e+05 | 7.525e+05 | False | 2 | 0.8602 | 0.8631 |
| telegrafico | char | 50% | 6.971e+05 | 7.333e+05 | False | 1 | 0.884 | 0.874 |
| telegrafico | char | 90% | NaN | NaN | False | 1 | NaN | NaN |

## D: vocabolario migliore per formato (params_total minimo al 90%, altrimenti al 50%)

| format | vocab migliore | livello | params_total | rapporto vs italiano_pieno migliore | rapporto vs italiano_pieno stesso vocab |
|---|---|---|---|---|---|
| interlingua_v1enc | bpe1024 | 90% | 7.059e+05 | 0.9247 | 0.8946 |
| italiano_pieno | bpe2048 | 90% | 7.634e+05 | 1 | 1 |
| telegrafico | bpe1024 | 90% | 6.904e+05 | 0.9044 | 0.875 |

## D: tempo di addestramento fino al 90% (train_time_s, log-log su params_total)

| format | vocab | params_total 90% | train_time_s 90% |
|---|---|---|---|
| interlingua_v1enc | bpe1024 | 7.059e+05 | 100.4 |
| interlingua_v1enc | bpe2048 | 7.448e+05 | 73.49 |
| interlingua_v1enc | bpe4096 | 8.656e+05 | 122.9 |
| interlingua_v1enc | bpe512 | 7.66e+05 | 118.9 |
| italiano_pieno | bpe1024 | 7.89e+05 | 135.1 |
| italiano_pieno | bpe2048 | 7.634e+05 | 117.6 |
| italiano_pieno | bpe4096 | 9.672e+05 | 173.7 |
| italiano_pieno | bpe512 | 8.718e+05 | 171.9 |
| telegrafico | bpe1024 | 6.904e+05 | 104.3 |
| telegrafico | bpe2048 | 7.591e+05 | 78.14 |
| telegrafico | bpe4096 | 8.722e+05 | 130.7 |
| telegrafico | bpe512 | 7.525e+05 | 131 |

## D_lr: lr_mult scelto per corpus e taglia

| corpus | size | lr_mult scelto | acc_mean | final_loss |
|---|---|---|---|---|
| interlingua_v1enc@bpe512 | 0.38M | 2 | 0.449 | 0.5501 |
| interlingua_v1enc@char | 0.38M | 4 | 0.1806 | 0.2954 |
| italiano_pieno@bpe512 | 0.38M | 2 | 0.3529 | 0.5552 |
| italiano_pieno@char | 0.38M | 4 | 0.1012 | 0.241 |
| telegrafico@bpe512 | 0.38M | 4 | 0.3764 | 0.5799 |
| telegrafico@char | 0.38M | 4 | 0.1238 | 0.2746 |

## D: latenza di risposta (media sui seed)

| format | vocab | size | batch | latency_ms_mean | ms_per_token | answers_per_s |
|---|---|---|---|---|---|---|
| interlingua_v1enc | bpe1024 | 0.15M | 1 | 1.971 | 0.4044 | 507.2 |
| interlingua_v1enc | bpe1024 | 0.15M | 64 | 2.667 | 0.547 | 5000 |
| interlingua_v1enc | bpe1024 | 0.20M | 1 | 2.351 | 0.4823 | 425.4 |
| interlingua_v1enc | bpe1024 | 0.20M | 64 | 3.636 | 0.7458 | 3667 |
| interlingua_v1enc | bpe1024 | 0.31M | 1 | 2.481 | 0.509 | 403 |
| interlingua_v1enc | bpe1024 | 0.31M | 64 | 3.197 | 0.6559 | 4170 |
| interlingua_v1enc | bpe1024 | 0.38M | 1 | 2.831 | 0.5807 | 353.3 |
| interlingua_v1enc | bpe1024 | 0.38M | 64 | 3.726 | 0.7644 | 3578 |
| interlingua_v1enc | bpe1024 | 0.60M | 1 | 2.409 | 0.4941 | 415.1 |
| interlingua_v1enc | bpe1024 | 0.60M | 64 | 3.323 | 0.6817 | 4012 |
| interlingua_v1enc | bpe1024 | 0.79M | 1 | 2.462 | 0.505 | 406.2 |
| interlingua_v1enc | bpe1024 | 0.79M | 64 | 3.509 | 0.7197 | 3800 |
| interlingua_v1enc | bpe2048 | 0.15M | 1 | 1.553 | 0.4326 | 644 |
| interlingua_v1enc | bpe2048 | 0.15M | 64 | 3.366 | 0.9378 | 4952 |
| interlingua_v1enc | bpe2048 | 0.20M | 1 | 1.964 | 0.5471 | 509.3 |
| interlingua_v1enc | bpe2048 | 0.20M | 64 | 3.582 | 0.9982 | 4652 |
| interlingua_v1enc | bpe2048 | 0.31M | 1 | 2.026 | 0.5646 | 493.5 |
| interlingua_v1enc | bpe2048 | 0.31M | 64 | 3.714 | 1.035 | 4487 |
| interlingua_v1enc | bpe2048 | 0.38M | 1 | 2.346 | 0.6537 | 426.2 |
| interlingua_v1enc | bpe2048 | 0.38M | 64 | 4.457 | 1.242 | 3740 |
| interlingua_v1enc | bpe2048 | 0.60M | 1 | 1.976 | 0.5505 | 506.1 |
| interlingua_v1enc | bpe2048 | 0.60M | 64 | 4.139 | 1.153 | 4026 |
| interlingua_v1enc | bpe2048 | 0.79M | 1 | 1.99 | 0.5546 | 502.4 |
| interlingua_v1enc | bpe2048 | 0.79M | 64 | 3.958 | 1.103 | 4211 |
| interlingua_v1enc | bpe4096 | 0.15M | 1 | 1.496 | 0.4863 | 668.3 |
| interlingua_v1enc | bpe4096 | 0.15M | 64 | 2.126 | 0.691 | 2.045e+04 |
| interlingua_v1enc | bpe4096 | 0.20M | 1 | 1.825 | 0.5931 | 548 |
| interlingua_v1enc | bpe4096 | 0.20M | 64 | 3.015 | 0.9798 | 1.442e+04 |
| interlingua_v1enc | bpe4096 | 0.31M | 1 | 1.887 | 0.6132 | 530 |
| interlingua_v1enc | bpe4096 | 0.31M | 64 | 2.614 | 0.8494 | 1.664e+04 |
| interlingua_v1enc | bpe4096 | 0.38M | 1 | 2.066 | 0.6715 | 484 |
| interlingua_v1enc | bpe4096 | 0.38M | 64 | 2.803 | 0.9109 | 1.551e+04 |
| interlingua_v1enc | bpe4096 | 0.60M | 1 | 1.782 | 0.5791 | 561.2 |
| interlingua_v1enc | bpe4096 | 0.60M | 64 | 2.517 | 0.8181 | 1.727e+04 |
| interlingua_v1enc | bpe4096 | 0.79M | 1 | 1.721 | 0.5592 | 581.1 |
| interlingua_v1enc | bpe4096 | 0.79M | 64 | 2.532 | 0.8227 | 1.717e+04 |
| interlingua_v1enc | bpe512 | 0.15M | 1 | 2.156 | 0.3233 | 463.7 |
| interlingua_v1enc | bpe512 | 0.15M | 64 | 2.987 | 0.4479 | 3071 |
| interlingua_v1enc | bpe512 | 0.20M | 1 | 2.852 | 0.4276 | 350.7 |
| interlingua_v1enc | bpe512 | 0.20M | 64 | 3.98 | 0.5968 | 2305 |
| interlingua_v1enc | bpe512 | 0.31M | 1 | 2.849 | 0.4272 | 351 |
| interlingua_v1enc | bpe512 | 0.31M | 64 | 3.816 | 0.5722 | 2404 |
| interlingua_v1enc | bpe512 | 0.38M | 1 | 3.242 | 0.4862 | 308.4 |
| interlingua_v1enc | bpe512 | 0.38M | 64 | 4.481 | 0.6719 | 2047 |
| interlingua_v1enc | bpe512 | 0.60M | 1 | 2.819 | 0.4227 | 354.7 |
| interlingua_v1enc | bpe512 | 0.60M | 64 | 4.233 | 0.6348 | 2167 |
| interlingua_v1enc | bpe512 | 0.79M | 1 | 2.759 | 0.4136 | 362.5 |
| interlingua_v1enc | bpe512 | 0.79M | 64 | 4.125 | 0.6185 | 2224 |
| interlingua_v1enc | char | 0.15M | 1 | 4.183 | 0.2654 | 239.1 |
| interlingua_v1enc | char | 0.15M | 64 | 5.221 | 0.3313 | 644.8 |
| interlingua_v1enc | char | 0.20M | 1 | 5.108 | 0.3241 | 195.8 |
| interlingua_v1enc | char | 0.20M | 64 | 6.443 | 0.4089 | 522.6 |
| interlingua_v1enc | char | 0.31M | 1 | 5.181 | 0.3288 | 193 |
| interlingua_v1enc | char | 0.31M | 64 | 6.659 | 0.4226 | 505.7 |
| interlingua_v1enc | char | 0.38M | 1 | 6.398 | 0.406 | 156.3 |
| interlingua_v1enc | char | 0.38M | 64 | 8.233 | 0.5224 | 409 |
| interlingua_v1enc | char | 0.60M | 1 | 5.317 | 0.3374 | 188.1 |
| interlingua_v1enc | char | 0.60M | 64 | 6.767 | 0.4294 | 497.6 |
| interlingua_v1enc | char | 0.79M | 1 | 5.162 | 0.3276 | 193.7 |
| interlingua_v1enc | char | 0.79M | 64 | 6.784 | 0.4305 | 496.3 |
| italiano_pieno | bpe1024 | 0.15M | 1 | 1.952 | 0.3909 | 512.2 |
| italiano_pieno | bpe1024 | 0.15M | 64 | 3.162 | 0.6331 | 3810 |
| italiano_pieno | bpe1024 | 0.20M | 1 | 2.292 | 0.4588 | 436.4 |
| italiano_pieno | bpe1024 | 0.20M | 64 | 3.541 | 0.7089 | 3403 |
| italiano_pieno | bpe1024 | 0.31M | 1 | 2.539 | 0.5083 | 393.9 |
| italiano_pieno | bpe1024 | 0.31M | 64 | 3.992 | 0.7993 | 3018 |
| italiano_pieno | bpe1024 | 0.38M | 1 | 2.757 | 0.5519 | 362.8 |
| italiano_pieno | bpe1024 | 0.38M | 64 | 4.056 | 0.8121 | 2970 |
| italiano_pieno | bpe1024 | 0.60M | 1 | 2.438 | 0.488 | 410.3 |
| italiano_pieno | bpe1024 | 0.60M | 64 | 3.768 | 0.7543 | 3198 |
| italiano_pieno | bpe1024 | 0.79M | 1 | 2.542 | 0.5088 | 393.5 |
| italiano_pieno | bpe1024 | 0.79M | 64 | 4.035 | 0.8077 | 2986 |
| italiano_pieno | bpe2048 | 0.15M | 1 | 1.614 | 0.4477 | 619.4 |
| italiano_pieno | bpe2048 | 0.15M | 64 | 2.563 | 0.7106 | 6846 |
| italiano_pieno | bpe2048 | 0.20M | 1 | 1.855 | 0.5145 | 539 |
| italiano_pieno | bpe2048 | 0.20M | 64 | 3.49 | 0.9678 | 5027 |
| italiano_pieno | bpe2048 | 0.31M | 1 | 2 | 0.5548 | 499.9 |
| italiano_pieno | bpe2048 | 0.31M | 64 | 3.292 | 0.913 | 5329 |
| italiano_pieno | bpe2048 | 0.38M | 1 | 2.357 | 0.6537 | 424.2 |
| italiano_pieno | bpe2048 | 0.38M | 64 | 3.865 | 1.072 | 4539 |
| italiano_pieno | bpe2048 | 0.60M | 1 | 2 | 0.5547 | 499.9 |
| italiano_pieno | bpe2048 | 0.60M | 64 | 3.212 | 0.8907 | 5462 |
| italiano_pieno | bpe2048 | 0.79M | 1 | 1.998 | 0.5541 | 500.4 |
| italiano_pieno | bpe2048 | 0.79M | 64 | 3.286 | 0.9114 | 5338 |
| italiano_pieno | bpe4096 | 0.15M | 1 | 1.478 | 0.4802 | 676.8 |
| italiano_pieno | bpe4096 | 0.15M | 64 | 1.975 | 0.642 | 2.025e+04 |
| italiano_pieno | bpe4096 | 0.20M | 1 | 1.425 | 0.4632 | 701.6 |
| italiano_pieno | bpe4096 | 0.20M | 64 | 1.923 | 0.6249 | 2.08e+04 |
| italiano_pieno | bpe4096 | 0.31M | 1 | 2.007 | 0.6521 | 498.4 |
| italiano_pieno | bpe4096 | 0.31M | 64 | 3.099 | 1.007 | 1.291e+04 |
| italiano_pieno | bpe4096 | 0.38M | 1 | 1.638 | 0.5324 | 610.4 |
| italiano_pieno | bpe4096 | 0.38M | 64 | 2.347 | 0.7627 | 1.704e+04 |
| italiano_pieno | bpe4096 | 0.60M | 1 | 1.839 | 0.5976 | 543.9 |
| italiano_pieno | bpe4096 | 0.60M | 64 | 2.609 | 0.8479 | 1.533e+04 |
| italiano_pieno | bpe4096 | 0.79M | 1 | 1.504 | 0.4886 | 665.1 |
| italiano_pieno | bpe4096 | 0.79M | 64 | 2.623 | 0.8523 | 1.525e+04 |
| italiano_pieno | bpe512 | 0.15M | 1 | 2.543 | 0.3385 | 393.2 |
| italiano_pieno | bpe512 | 0.15M | 64 | 3.899 | 0.5189 | 2120 |
| italiano_pieno | bpe512 | 0.20M | 1 | 3.148 | 0.4191 | 317.6 |
| italiano_pieno | bpe512 | 0.20M | 64 | 4.25 | 0.5656 | 1945 |
| italiano_pieno | bpe512 | 0.31M | 1 | 3.109 | 0.4138 | 321.7 |
| italiano_pieno | bpe512 | 0.31M | 64 | 4.451 | 0.5925 | 1857 |
| italiano_pieno | bpe512 | 0.38M | 1 | 3.561 | 0.474 | 280.8 |
| italiano_pieno | bpe512 | 0.38M | 64 | 5.134 | 0.6834 | 1610 |
| italiano_pieno | bpe512 | 0.60M | 1 | 3.171 | 0.4221 | 315.4 |
| italiano_pieno | bpe512 | 0.60M | 64 | 4.177 | 0.556 | 1979 |
| italiano_pieno | bpe512 | 0.79M | 1 | 3.205 | 0.4267 | 312 |
| italiano_pieno | bpe512 | 0.79M | 64 | 4.225 | 0.5623 | 1956 |
| italiano_pieno | char | 0.15M | 1 | 4.705 | 0.2986 | 212.5 |
| italiano_pieno | char | 0.15M | 64 | 5.735 | 0.364 | 484.3 |
| italiano_pieno | char | 0.20M | 1 | 5.874 | 0.3728 | 170.2 |
| italiano_pieno | char | 0.20M | 64 | 6.938 | 0.4403 | 400.4 |
| italiano_pieno | char | 0.31M | 1 | 5.839 | 0.3706 | 171.2 |
| italiano_pieno | char | 0.31M | 64 | 7.059 | 0.448 | 393.5 |
| italiano_pieno | char | 0.38M | 1 | 7.218 | 0.458 | 138.5 |
| italiano_pieno | char | 0.38M | 64 | 8.654 | 0.5492 | 321 |
| italiano_pieno | char | 0.60M | 1 | 5.968 | 0.3787 | 167.6 |
| italiano_pieno | char | 0.60M | 64 | 7.506 | 0.4763 | 370.1 |
| italiano_pieno | char | 0.79M | 1 | 5.973 | 0.379 | 167.4 |
| italiano_pieno | char | 0.79M | 64 | 7.08 | 0.4493 | 392.4 |
| telegrafico | bpe1024 | 0.15M | 1 | 1.967 | 0.4021 | 508.5 |
| telegrafico | bpe1024 | 0.15M | 64 | 2.608 | 0.5331 | 5401 |
| telegrafico | bpe1024 | 0.20M | 1 | 2.324 | 0.4752 | 430.3 |
| telegrafico | bpe1024 | 0.20M | 64 | 3.865 | 0.7902 | 3644 |
| telegrafico | bpe1024 | 0.31M | 1 | 2.358 | 0.4821 | 424.1 |
| telegrafico | bpe1024 | 0.31M | 64 | 3.33 | 0.6809 | 4229 |
| telegrafico | bpe1024 | 0.38M | 1 | 2.82 | 0.5765 | 354.7 |
| telegrafico | bpe1024 | 0.38M | 64 | 4.866 | 0.9948 | 2895 |
| telegrafico | bpe1024 | 0.60M | 1 | 2.45 | 0.5009 | 408.2 |
| telegrafico | bpe1024 | 0.60M | 64 | 3.38 | 0.691 | 4168 |
| telegrafico | bpe1024 | 0.79M | 1 | 2.407 | 0.4921 | 415.5 |
| telegrafico | bpe1024 | 0.79M | 64 | 3.354 | 0.6857 | 4200 |
| telegrafico | bpe2048 | 0.15M | 1 | 1.562 | 0.434 | 640 |
| telegrafico | bpe2048 | 0.15M | 64 | 2.531 | 0.7029 | 6813 |
| telegrafico | bpe2048 | 0.20M | 1 | 1.973 | 0.548 | 506.9 |
| telegrafico | bpe2048 | 0.20M | 64 | 2.689 | 0.7469 | 6413 |
| telegrafico | bpe2048 | 0.31M | 1 | 1.95 | 0.5416 | 512.9 |
| telegrafico | bpe2048 | 0.31M | 64 | 3.065 | 0.8513 | 5626 |
| telegrafico | bpe2048 | 0.38M | 1 | 2.33 | 0.6471 | 429.3 |
| telegrafico | bpe2048 | 0.38M | 64 | 3.298 | 0.9162 | 5227 |
| telegrafico | bpe2048 | 0.60M | 1 | 1.869 | 0.5192 | 535 |
| telegrafico | bpe2048 | 0.60M | 64 | 2.835 | 0.7876 | 6081 |
| telegrafico | bpe2048 | 0.79M | 1 | 2.188 | 0.6078 | 457 |
| telegrafico | bpe2048 | 0.79M | 64 | 2.854 | 0.7927 | 6042 |
| telegrafico | bpe4096 | 0.15M | 1 | 1.468 | 0.477 | 681.3 |
| telegrafico | bpe4096 | 0.15M | 64 | 2.328 | 0.7565 | 1.868e+04 |
| telegrafico | bpe4096 | 0.20M | 1 | 1.84 | 0.5979 | 543.5 |
| telegrafico | bpe4096 | 0.20M | 64 | 2.265 | 0.7361 | 1.92e+04 |
| telegrafico | bpe4096 | 0.31M | 1 | 1.803 | 0.5861 | 554.5 |
| telegrafico | bpe4096 | 0.31M | 64 | 2.324 | 0.7553 | 1.871e+04 |
| telegrafico | bpe4096 | 0.38M | 1 | 2.166 | 0.7039 | 461.7 |
| telegrafico | bpe4096 | 0.38M | 64 | 2.73 | 0.8873 | 1.592e+04 |
| telegrafico | bpe4096 | 0.60M | 1 | 1.765 | 0.5735 | 566.7 |
| telegrafico | bpe4096 | 0.60M | 64 | 2.443 | 0.794 | 1.78e+04 |
| telegrafico | bpe4096 | 0.79M | 1 | 1.717 | 0.5579 | 582.5 |
| telegrafico | bpe4096 | 0.79M | 64 | 2.454 | 0.7975 | 1.772e+04 |
| telegrafico | bpe512 | 0.15M | 1 | 2.343 | 0.3381 | 426.9 |
| telegrafico | bpe512 | 0.15M | 64 | 3.542 | 0.5112 | 2639 |
| telegrafico | bpe512 | 0.20M | 1 | 3.075 | 0.4438 | 325.3 |
| telegrafico | bpe512 | 0.20M | 64 | 4.472 | 0.6454 | 2090 |
| telegrafico | bpe512 | 0.31M | 1 | 2.889 | 0.417 | 346.1 |
| telegrafico | bpe512 | 0.31M | 64 | 3.978 | 0.5742 | 2349 |
| telegrafico | bpe512 | 0.38M | 1 | 3.526 | 0.5089 | 283.6 |
| telegrafico | bpe512 | 0.38M | 64 | 5.06 | 0.7304 | 1847 |
| telegrafico | bpe512 | 0.60M | 1 | 2.982 | 0.4305 | 335.3 |
| telegrafico | bpe512 | 0.60M | 64 | 4.526 | 0.6532 | 2065 |
| telegrafico | bpe512 | 0.79M | 1 | 2.943 | 0.4248 | 339.8 |
| telegrafico | bpe512 | 0.79M | 64 | 4.35 | 0.6278 | 2149 |
| telegrafico | char | 0.15M | 1 | 4.291 | 0.2723 | 233.1 |
| telegrafico | char | 0.15M | 64 | 5.571 | 0.3535 | 612.7 |
| telegrafico | char | 0.20M | 1 | 5.408 | 0.3432 | 184.9 |
| telegrafico | char | 0.20M | 64 | 6.794 | 0.4311 | 502.4 |
| telegrafico | char | 0.31M | 1 | 5.523 | 0.3505 | 181.1 |
| telegrafico | char | 0.31M | 64 | 7.13 | 0.4525 | 478.7 |
| telegrafico | char | 0.38M | 1 | 6.532 | 0.4145 | 153.1 |
| telegrafico | char | 0.38M | 64 | 8.601 | 0.5458 | 396.8 |
| telegrafico | char | 0.60M | 1 | 5.433 | 0.3448 | 184.1 |
| telegrafico | char | 0.60M | 64 | 7.1 | 0.4505 | 480.7 |
| telegrafico | char | 0.79M | 1 | 5.459 | 0.3464 | 183.2 |
| telegrafico | char | 0.79M | 64 | 7.16 | 0.4544 | 476.7 |

## D: tempi di passo (testa e corpo, media)

| format | vocab | tokens_step | T | head_ms | body_ms |
|---|---|---|---|---|---|
| interlingua_v1enc | bpe1024 | 23816 | 70 | 2.981 | 11.44 |
| interlingua_v1enc | bpe2048 | 19420 | 52 | 4.355 | 7.675 |
| interlingua_v1enc | bpe4096 | 16020 | 40 | 6.574 | 6.252 |
| interlingua_v1enc | bpe512 | 30023 | 88 | 2.299 | 13.9 |
| interlingua_v1enc | char | 71083 | 189 | 1.729 | 36.27 |
| italiano_pieno | bpe1024 | 34610 | 93 | 1.417 | 12.47 |
| italiano_pieno | bpe2048 | 29957 | 78 | 2.084 | 11.34 |
| italiano_pieno | bpe4096 | 26500 | 64 | 10.73 | 7.831 |
| italiano_pieno | bpe512 | 43297 | 120 | 1.58 | 14.79 |
| italiano_pieno | char | 118644 | 303 | 0.912 | 53.08 |
| telegrafico | bpe1024 | 25890 | 74 | 3.266 | 10.94 |
| telegrafico | bpe2048 | 21416 | 56 | 4.824 | 8.875 |
| telegrafico | bpe4096 | 18068 | 44 | 7.395 | 6.666 |
| telegrafico | bpe512 | 32994 | 98 | 2.531 | 13.18 |
| telegrafico | char | 87735 | 225 | 2.614 | 37.3 |

## E: telegrafico_auto vs telegrafico (v2, bpe4096)

| corpus | livello | params_nonemb | params_total | lower_bound | n_seeds | rapporto_nonemb_vs_telegrafico | rapporto_total_vs_telegrafico |
|---|---|---|---|---|---|---|---|
| telegrafico | 50% | 2.185e+05 | 4.749e+05 | False | 2 | 1 | 1 |
| telegrafico | 90% | 5.015e+05 | 8.722e+05 | False | 2 | 1 | 1 |
| telegrafico_auto | 50% | 2.606e+05 | 5.521e+05 | False | 2 | 1.192 | 1.163 |
| telegrafico_auto | 90% | 5.719e+05 | 9.972e+05 | False | 2 | 1.14 | 1.143 |

## E: telegrafico_lista, accuratezza per taglia

| size | params_nonemb | acc_mean | n_seeds |
|---|---|---|---|
| 0.60M | 6.082e+05 | 0.9755 | 1 |
| 1.49M | 1.496e+06 | 0.9961 | 1 |
| 4M | 3.997e+06 | 0.9963 | 1 |

## E: confronto stripper (medie)

| pieno | spacy | lista | solo_spacy | parole_pieno | parole_spacy | parole_lista | solo_lista |
|---|---|---|---|---|---|---|---|
| Abele Ferraro ha i natali il 3 settembre 1981. Ha scelto il percorso di Agraria. Ha studiato presso Università di Venezia. Lavora come statistico. Ha lavorato presso Ferretti Yachts. Lui è originario di Cuneo. | Abele Ferraro ha natali 3 settembre 1981. scelto percorso Agraria. studiato Università Venezia. Lavora statistico. lavorato Ferretti Yachts. originario Cuneo. | Abele Ferraro natali 3 settembre 1981. scelto percorso Agraria. studiato Università Venezia. Lavora come statistico. lavorato Ferretti Yachts. originario Cuneo. | come | 34 | 20 | 20 | NaN |
| Adele Bertelli ha i natali a Fano. Ha i natali il 11 aprile 1998. Lei collabora con Technogym. Lei lavora come ottico. Lei ha frequentato Università della Calabria. Ha conseguito il titolo in Logopedia. | Adele Bertelli ha natali Fano. Ha natali 11 aprile 1998. collabora Technogym. lavora ottico. frequentato Università Calabria. conseguito titolo Logopedia. | Adele Bertelli natali Fano. natali 11 aprile 1998. collabora Technogym. lavora come ottico. frequentato Università Calabria. conseguito titolo Logopedia. | come | 34 | 20 | 19 | NaN |
| Aldo Mancini ha studiato presso Istituto Universitario di Architettura di Venezia. Lavora come statistico. Ha visto la luce a Anzio. Ha scelto il percorso di Management. È impiegato presso Brioni. Lui ha visto la luce il 29 ottobre 1993. | Aldo Mancini studiato Istituto Universitario Architettura Venezia. Lavora statistico. visto luce Anzio. scelto percorso Management. impiegato Brioni. visto luce 29 ottobre 1993. | Aldo Mancini studiato Istituto Universitario Architettura Venezia. Lavora come statistico. visto luce Anzio. scelto percorso Management. impiegato Brioni. visto luce 29 ottobre 1993. | come | 39 | 22 | 23 | NaN |
| Alessandro Boccia è originario di Bisceglie. È impiegato presso Lavazza. Lui è barbiere. Ha come data di nascita il 17 agosto 1975. Lui ha studiato presso Università di Chieti-Pescara. Lui ha seguito il corso di Bioinformatica. | Alessandro Boccia originario Bisceglie. impiegato Lavazza. barbiere. Ha data nascita 17 agosto 1975. studiato Università Chieti-Pescara. seguito corso Bioinformatica. | Alessandro Boccia originario Bisceglie. impiegato Lavazza. barbiere. come data nascita 17 agosto 1975. studiato Università Chieti- Pescara. seguito corso Bioinformatica. | come | 36 | 19 | 20 | NaN |
| Alessandro Cacciola ha conseguito la laurea presso Università di Roma Tor Vergata. Lui lavora per Comau. Lavora come agronomo. Lui ha conseguito il titolo in Lingue e letterature straniere. Lui ha visto la luce il 29 settembre 1951. Lui è nato a Reggio Calabria. | Alessandro Cacciola conseguito laurea Università Roma Tor Vergata. lavora Comau. Lavora agronomo. conseguito titolo Lingue e letterature straniere. visto luce 29 settembre 1951. nato Reggio Calabria. | Alessandro Cacciola conseguito laurea Università Roma Tor Vergata. lavora Comau. Lavora come agronomo. conseguito titolo Lingue e letterature straniere. visto luce 29 settembre 1951. nato Reggio Calabria. | come | 44 | 26 | 27 | NaN |
| Annalisa Fabbri lavora per Barilla. Lei lavora come insegnante. Ha visto la luce il 29 novembre 1984. Lei ha conseguito la laurea presso Università Bocconi. Ha visto la luce a Afragola. Ha scelto il percorso di Scienze politiche. | Annalisa Fabbri lavora Barilla. lavora insegnante. visto luce 29 novembre 1984. conseguito laurea Università Bocconi. visto luce Afragola. scelto percorso Scienze politiche. | Annalisa Fabbri lavora Barilla. lavora come insegnante. visto luce 29 novembre 1984. conseguito laurea Università Bocconi. visto luce Afragola. scelto percorso Scienze politiche. | come | 38 | 22 | 23 | NaN |
| Anselmo Bertocchi ha come data di nascita il 21 febbraio 1964. Lui collabora con Oviesse. Lui ha il ruolo di sistemista. Lui è originario di Campobasso. Ha seguito il corso di Bioinformatica. Ha studiato presso Università di Verona Scuola Medica. | Anselmo Bertocchi ha data nascita 21 febbraio 1964. collabora Oviesse. ha ruolo sistemista. originario Campobasso. seguito corso Bioinformatica. studiato Università Verona Scuola Medica. | Anselmo Bertocchi come data nascita 21 febbraio 1964. collabora Oviesse. ruolo sistemista. originario Campobasso. seguito corso Bioinformatica. studiato Università Verona Scuola Medica. | come | 40 | 23 | 22 | NaN |
| Anselmo Coppola è nato a Corato. Ha conseguito la laurea presso Università di Torino Scuola di Medicina. Ha il ruolo di illustratore. Ha conseguito il titolo in Fisica. Lui ha come data di nascita il 21 agosto 2002. Collabora con Dedagroup. | Anselmo Coppola nato Corato. conseguito laurea Università Torino Scuola Medicina. Ha ruolo illustratore. conseguito titolo Fisica. ha data nascita 21 agosto 2002. Collabora Dedagroup. | Anselmo Coppola nato Corato. conseguito laurea Università Torino Scuola Medicina. ruolo illustratore. conseguito titolo Fisica. come data nascita 21 agosto 2002. Collabora Dedagroup. | come | 41 | 24 | 23 | NaN |
| Antonio Ferrari è nato a Livorno. Ha come data di nascita il 24 aprile 1977. Lui ha conseguito la laurea presso Università di Bari Scuola di Medicina. Lavora come storico. È impiegato presso Esselunga. Lui ha seguito il corso di Antropologia culturale. | Antonio Ferrari nato Livorno. Ha data nascita 24 aprile 1977. conseguito laurea Università Bari Scuola Medicina. Lavora storico. impiegato Esselunga. seguito corso Antropologia culturale. | Antonio Ferrari nato Livorno. come data nascita 24 aprile 1977. conseguito laurea Università Bari Scuola Medicina. Lavora come storico. impiegato Esselunga. seguito corso Antropologia culturale. | come | 42 | 24 | 25 | NaN |
| Assunta Parisi lavora come pescatore. Lei ha come data di nascita il 6 luglio 1973. Lei lavora per Fastweb. Ha frequentato Università di Messina. Lei ha visto la luce a Sulmona. Lei ha seguito il corso di Ingegneria elettrica. | Assunta Parisi lavora pescatore. ha data nascita 6 luglio 1973. lavora Fastweb. frequentato Università Messina. visto luce Sulmona. seguito corso Ingegneria elettrica. | Assunta Parisi lavora come pescatore. come data nascita 6 luglio 1973. lavora Fastweb. frequentato Università Messina. visto luce Sulmona. seguito corso Ingegneria elettrica. | come | 39 | 22 | 23 | NaN |
| Attilio Mazza lavora come tornitore. Ha visto la luce il 8 agosto 1965. Ha conseguito il titolo in Medicina e chirurgia. Lavora per Zanussi. Lui ha studiato presso Politecnico di Torino. È originario di Pordenone. | Attilio Mazza lavora tornitore. visto luce 8 agosto 1965. conseguito titolo Medicina e chirurgia. Lavora Zanussi. studiato Politecnico Torino. originario Pordenone. | Attilio Mazza lavora come tornitore. visto luce 8 agosto 1965. conseguito titolo Medicina e chirurgia. Lavora Zanussi. studiato Politecnico Torino. originario Pordenone. | come | 35 | 21 | 22 | NaN |
| Barbara Cescon è nata a Imola. Lavora come agronomo. Ha scelto il percorso di Ostetricia. Ha studiato presso Università di Foggia Scuola di Medicina. Lei ha lavorato presso De Longhi. È nata il 13 dicembre 1999. | Barbara Cescon nata Imola. Lavora agronomo. scelto percorso Ostetricia. studiato Università Foggia Scuola Medicina. lavorato De Longhi. nata 13 dicembre 1999. | Barbara Cescon nata Imola. Lavora come agronomo. scelto percorso Ostetricia. studiato Università Foggia Scuola Medicina. lavorato De Longhi. nata 13 dicembre 1999. | come | 36 | 21 | 22 | NaN |
| Benedetto Pavan si è laureato in Scienze biologiche. Lui ha i natali il 26 ottobre 1963. Lavora come illustratore. Lui ha visto la luce a Comacchio. È impiegato presso Unipol. Ha fatto gli studi presso Università di Pisa Scuola Sant Anna. | Benedetto Pavan si laureato Scienze biologiche. ha natali 26 ottobre 1963. Lavora illustratore. visto luce Comacchio. impiegato Unipol. fatto studi Università Pisa Scuola Sant Anna. | Benedetto Pavan si laureato Scienze biologiche. natali 26 ottobre 1963. Lavora come illustratore. visto luce Comacchio. impiegato Unipol. fatto studi Università Pisa Scuola Sant Anna. | come | 41 | 25 | 25 | NaN |
| Bianca Spada ha studiato presso Università di Verona. Ha i natali il 13 dicembre 1974. Lei ha visto la luce a Sora. Ha scelto il percorso di Archeologia. Lavora come redattore. Lei collabora con Natuzzi Group. | Bianca Spada studiato Università Verona. Ha natali 13 dicembre 1974. visto luce Sora. scelto percorso Archeologia. Lavora redattore. collabora Natuzzi Group. | Bianca Spada studiato Università Verona. natali 13 dicembre 1974. visto luce Sora. scelto percorso Archeologia. Lavora come redattore. collabora Natuzzi Group. | come | 36 | 21 | 21 | NaN |
| Carlo Bonomi ha fatto gli studi presso Università di Pavia. Ha conseguito il titolo in Diritto penale. Ha come data di nascita il 3 agosto 1993. Lui è impiegato presso Conad. È nato a Pavia. Lui esercita la professione di storico. | Carlo Bonomi fatto studi Università Pavia. conseguito titolo Diritto penale. Ha data nascita 3 agosto 1993. impiegato Conad. nato Pavia. esercita professione storico. | Carlo Bonomi fatto studi Università Pavia. conseguito titolo Diritto penale. come data nascita 3 agosto 1993. impiegato Conad. nato Pavia. esercita professione storico. | come | 41 | 23 | 23 | NaN |
| Carmela Liguori è impiegata presso Missoni. Lei ha come data di nascita il 14 aprile 1983. Lei ha seguito il corso di Psicologia clinica. È originaria di Trapani. Lei lavora come filosofo. Ha frequentato Università di Salerno Scuola di Medicina. | Carmela Liguori impiegata Missoni. ha data nascita 14 aprile 1983. seguito corso Psicologia clinica. originaria Trapani. lavora filosofo. frequentato Università Salerno Scuola Medicina. | Carmela Liguori impiegata Missoni. come data nascita 14 aprile 1983. seguito corso Psicologia clinica. originaria Trapani. lavora come filosofo. frequentato Università Salerno Scuola Medicina. | come | 40 | 23 | 24 | NaN |
| Cecilia Alberti ha studiato presso Università di Pavia Collegio Universitario. Lei lavora come osteopata. Ha conseguito il titolo in Cybersicurezza. È originaria di Trieste. Lei ha i natali il 16 maggio 2004. Lei è impiegata presso Luxottica. | Cecilia Alberti studiato Università Pavia Collegio Universitario. lavora osteopata. conseguito titolo Cybersicurezza. originaria Trieste. ha natali 16 maggio 2004. impiegata Luxottica. | Cecilia Alberti studiato Università Pavia Collegio Universitario. lavora come osteopata. conseguito titolo Cybersicurezza. originaria Trieste. natali 16 maggio 2004. impiegata Luxottica. | come | 37 | 21 | 21 | NaN |
| Cesare Montanari ha conseguito il titolo in Tecnologie alimentari. Lui lavora come commerciante. Lui è originario di Acerra. Lui ha fatto gli studi presso Università di Ferrara. Lui ha come data di nascita il 16 luglio 1968. Lui è impiegato presso Pirelli. | Cesare Montanari conseguito titolo Tecnologie alimentari. lavora commerciante. originario Acerra. fatto studi Università Ferrara. ha data nascita 16 luglio 1968. impiegato Pirelli. | Cesare Montanari conseguito titolo Tecnologie alimentari. lavora come commerciante. originario Acerra. fatto studi Università Ferrara. come data nascita 16 luglio 1968. impiegato Pirelli. | come | 42 | 22 | 23 | NaN |
| Cesare Ranieri si è laureato in Scienze infermieristiche pediatriche. Lavora come traduttore. Ha conseguito la laurea presso Università di Napoli Federico II. Ha i natali a Sulmona. È nato il 22 settembre 1987. È impiegato presso Fastweb. | Cesare Ranieri si laureato Scienze infermieristiche pediatriche. Lavora traduttore. conseguito laurea Università Napoli Federico II. Ha natali Sulmona. nato 22 settembre 1987. impiegato Fastweb. | Cesare Ranieri si laureato Scienze infermieristiche pediatriche. Lavora come traduttore. conseguito laurea Università Napoli Federico II. natali Sulmona. nato 22 settembre 1987. impiegato Fastweb. | come | 37 | 24 | 24 | NaN |
| Chiara Marino ha studiato presso Università di Trento Scuola di Ingegneria. È originaria di Bari. Lavora come medico. Lei ha lavorato presso Carapelli. Ha i natali il 22 giugno 1975. Lei ha scelto il percorso di Ingegneria dell autoveicolo. | Chiara Marino studiato Università Trento Scuola Ingegneria. originaria Bari. Lavora medico. lavorato Carapelli. Ha natali 22 giugno 1975. scelto percorso Ingegneria autoveicolo. | Chiara Marino studiato Università Trento Scuola Ingegneria. originaria Bari. Lavora come medico. lavorato Carapelli. natali 22 giugno 1975. scelto percorso Ingegneria dell autoveicolo. | come dell | 39 | 22 | 23 | NaN |
| Cristina Lanza si è laureata in Bioinformatica. Lei lavora per Natuzzi. Lavora come mediatore. Lei ha i natali il 23 settembre 2001. Lei è nata a Milano. Lei ha studiato presso Università di Urbino. | Cristina Lanza si laureata Bioinformatica. lavora Natuzzi. Lavora mediatore. ha natali 23 settembre 2001. nata Milano. studiato Università Urbino. | Cristina Lanza si laureata Bioinformatica. lavora Natuzzi. Lavora come mediatore. natali 23 settembre 2001. nata Milano. studiato Università Urbino. | come | 34 | 19 | 19 | NaN |
| Daniele Ruggiero ha frequentato Politecnico di Torino. Ha i natali a Afragola. Lui lavora per Scavolini. Ha conseguito il titolo in Storia dell architettura. Lui ha i natali il 12 novembre 1984. Lavora come camionista. | Daniele Ruggiero frequentato Politecnico Torino. Ha natali Afragola. lavora Scavolini. conseguito titolo Storia architettura. ha natali 12 novembre 1984. Lavora camionista. | Daniele Ruggiero frequentato Politecnico Torino. natali Afragola. lavora Scavolini. conseguito titolo Storia dell architettura. natali 12 novembre 1984. Lavora come camionista. | come dell | 35 | 21 | 21 | NaN |
| Davide Parisi ha conseguito il titolo in Economia del turismo. Lui è nato a Enna. Esercita la professione di economista. Lui ha come data di nascita il 18 dicembre 1980. Lui ha fatto gli studi presso Università di Ferrara. Lavora per Prada. | Davide Parisi conseguito titolo Economia turismo. nato Enna. Esercita professione economista. ha data nascita 18 dicembre 1980. fatto studi Università Ferrara. Lavora Prada. | Davide Parisi conseguito titolo Economia turismo. nato Enna. Esercita professione economista. come data nascita 18 dicembre 1980. fatto studi Università Ferrara. Lavora Prada. | come | 42 | 23 | 23 | NaN |
| Donato Massa ha conseguito il titolo in Enologia. Lavora come psicologo. Ha visto la luce il 23 dicembre 2004. Lui ha lavorato presso Benetton. Lui è nato a Gravina in Puglia. Lui ha fatto gli studi presso Università Telematica Internazionale UNINETTUNO. | Donato Massa conseguito titolo Enologia. Lavora psicologo. visto luce 23 dicembre 2004. lavorato Benetton. nato Gravina Puglia. fatto studi Università Telematica Internazionale UNINETTUNO. | Donato Massa conseguito titolo Enologia. Lavora come psicologo. visto luce 23 dicembre 2004. lavorato Benetton. nato Gravina Puglia. fatto studi Università Telematica Internazionale UNINETTUNO. | come | 41 | 23 | 24 | NaN |
| Dora Costa ha come data di nascita il 30 marzo 1972. È originaria di Trento. Si è laureata in Ingegneria aerospaziale. Ha fatto gli studi presso Università Luiss Guido Carli. Lei lavora come professore. È impiegata presso Rizzoli. | Dora Costa ha data nascita 30 marzo 1972. originaria Trento. Si laureata Ingegneria aerospaziale. fatto studi Università Luiss Guido Carli. lavora professore. impiegata Rizzoli. | Dora Costa come data nascita 30 marzo 1972. originaria Trento. Si laureata Ingegneria aerospaziale. fatto studi Università Luiss Guido Carli. lavora come professore. impiegata Rizzoli. | come | 38 | 24 | 25 | NaN |
| Duilio Brambilla lavora per Smeg. Lui è nato il 5 agosto 1984. Ha studiato presso Università di Bari Scuola di Medicina. Lavora come psicologo. Ha conseguito il titolo in Veterinaria. Lui ha i natali a Aosta. | Duilio Brambilla lavora Smeg. nato 5 agosto 1984. studiato Università Bari Scuola Medicina. Lavora psicologo. conseguito titolo Veterinaria. ha natali Aosta. | Duilio Brambilla lavora Smeg. nato 5 agosto 1984. studiato Università Bari Scuola Medicina. Lavora come psicologo. conseguito titolo Veterinaria. natali Aosta. | come | 36 | 21 | 21 | NaN |
| Edoardo Franceschini ha scelto il percorso di Scienze pedagogiche. È impiegato presso Eurospin. Lui ha i natali il 24 febbraio 1967. Ha i natali a Agrigento. Lui lavora come musicista. Lui ha studiato presso Università della Basilicata. | Edoardo Franceschini scelto percorso Scienze pedagogiche. impiegato Eurospin. ha natali 24 febbraio 1967. Ha natali Agrigento. lavora musicista. studiato Università Basilicata. | Edoardo Franceschini scelto percorso Scienze pedagogiche. impiegato Eurospin. natali 24 febbraio 1967. natali Agrigento. lavora come musicista. studiato Università Basilicata. | come | 37 | 21 | 20 | NaN |
| Elisabetta Pinto ha come data di nascita il 19 novembre 1979. Lei ha frequentato Università di Roma Foro Italico. Lei esercita la professione di tornitore. Lei è impiegata presso Armani. Lei ha visto la luce a Nettuno. Ha conseguito il titolo in Chimica. | Elisabetta Pinto ha data nascita 19 novembre 1979. frequentato Università Roma Foro Italico. esercita professione tornitore. impiegata Armani. visto luce Nettuno. conseguito titolo Chimica. | Elisabetta Pinto come data nascita 19 novembre 1979. frequentato Università Roma Foro Italico. esercita professione tornitore. impiegata Armani. visto luce Nettuno. conseguito titolo Chimica. | come | 43 | 24 | 24 | NaN |
| Emilia Mattei lavora come tipografo. Ha i natali il 26 gennaio 1955. Lei collabora con Intimissimi. È nata a Faenza. Lei si è laureata in Ingegneria idraulica. Lei ha fatto gli studi presso Università Luiss Guido Carli. | Emilia Mattei lavora tipografo. Ha natali 26 gennaio 1955. collabora Intimissimi. nata Faenza. si laureata Ingegneria idraulica. fatto studi Università Luiss Guido Carli. | Emilia Mattei lavora come tipografo. natali 26 gennaio 1955. collabora Intimissimi. nata Faenza. si laureata Ingegneria idraulica. fatto studi Università Luiss Guido Carli. | come | 37 | 23 | 23 | NaN |
| Erika Sini ha i natali a Monfalcone. Ha frequentato Università di Sassari Scuola di Agraria. Lei ha come data di nascita il 22 gennaio 1998. Ha conseguito il titolo in Pedagogia. Lavora per Webuild. Lei esercita la professione di anestesista. | Erika Sini ha natali Monfalcone. frequentato Università Sassari Scuola Agraria. ha data nascita 22 gennaio 1998. conseguito titolo Pedagogia. Lavora Webuild. esercita professione anestesista. | Erika Sini natali Monfalcone. frequentato Università Sassari Scuola Agraria. come data nascita 22 gennaio 1998. conseguito titolo Pedagogia. Lavora Webuild. esercita professione anestesista. | come | 40 | 24 | 23 | NaN |
| Fabio Agostini è impiegato presso A2A. È nato a Siracusa. Lui è nato il 23 agosto 1993. Lui ha seguito il corso di Statistica. Lui ha frequentato Università Vita-Salute San Raffaele. Lavora come dentista. | Fabio Agostini impiegato A2A. nato Siracusa. nato 23 agosto 1993. seguito corso Statistica. frequentato Università Vita-Salute San Raffaele. Lavora dentista. | Fabio Agostini impiegato A2A. nato Siracusa. nato 23 agosto 1993. seguito corso Statistica. frequentato Università Vita- Salute San Raffaele. Lavora come dentista. | come | 34 | 20 | 22 | NaN |
| Fiorenzo Granata ha lavorato presso Natuzzi Group. Lui ha conseguito il titolo in Scienze geologiche. Ha il ruolo di giardiniere. Lui ha studiato presso Università di Genova Scuola di Scienze. Lui ha come data di nascita il 7 luglio 2000. Ha i natali a Lucca. | Fiorenzo Granata lavorato Natuzzi Group. conseguito titolo Scienze geologiche. Ha ruolo giardiniere. studiato Università Genova Scuola Scienze. ha data nascita 7 luglio 2000. Ha natali Lucca. | Fiorenzo Granata lavorato Natuzzi Group. conseguito titolo Scienze geologiche. ruolo giardiniere. studiato Università Genova Scuola Scienze. come data nascita 7 luglio 2000. natali Lucca. | come | 45 | 26 | 24 | NaN |
| Gaia Meli ha visto la luce il 18 novembre 1999. È nata a Mirandola. Lei ha frequentato Università di Teramo Scuola di Scienze. Lei ha seguito il corso di Geografia. Lavora come promotore. Lei ha lavorato presso Luceplan. | Gaia Meli visto luce 18 novembre 1999. nata Mirandola. frequentato Università Teramo Scuola Scienze. seguito corso Geografia. Lavora promotore. lavorato Luceplan. | Gaia Meli visto luce 18 novembre 1999. nata Mirandola. frequentato Università Teramo Scuola Scienze. seguito corso Geografia. Lavora come promotore. lavorato Luceplan. | come | 38 | 21 | 22 | NaN |
| Gennaro Tedeschi ha frequentato Università di Pisa Scuola Sant Anna. Lui lavora come falegname. Lui è nato il 9 febbraio 1997. È nato a Milano. Si è laureato in Relazioni internazionali. Lui lavora per Smeg Italia. | Gennaro Tedeschi frequentato Università Pisa Scuola Sant Anna. lavora falegname. nato 9 febbraio 1997. nato Milano. Si laureato Relazioni internazionali. lavora Smeg Italia. | Gennaro Tedeschi frequentato Università Pisa Scuola Sant Anna. lavora come falegname. nato 9 febbraio 1997. nato Milano. Si laureato Relazioni internazionali. lavora Smeg Italia. | come | 36 | 23 | 24 | NaN |
| Giada Palladino ha i natali a Gravina in Puglia. Lei è falegname. Lei ha frequentato Università di Brescia Scuola di Medicina. Lei ha seguito il corso di Lingue e letterature straniere. Ha come data di nascita il 4 marzo 1976. È impiegata presso Peroni. | Giada Palladino ha natali Gravina Puglia. falegname. frequentato Università Brescia Scuola Medicina. seguito corso Lingue e letterature straniere. Ha data nascita 4 marzo 1976. impiegata Peroni. | Giada Palladino natali Gravina Puglia. falegname. frequentato Università Brescia Scuola Medicina. seguito corso Lingue e letterature straniere. come data nascita 4 marzo 1976. impiegata Peroni. | come | 44 | 26 | 25 | NaN |
| Giada Piras è nata il 11 marzo 1969. Ha lavorato presso Biesse. È nata a Macerata. Ha conseguito il titolo in Ingegneria elettrica. Lei lavora come commerciante. Lei ha conseguito la laurea presso Università Vita-Salute San Raffaele. | Giada Piras nata 11 marzo 1969. lavorato Biesse. nata Macerata. conseguito titolo Ingegneria elettrica. lavora commerciante. conseguito laurea Università Vita-Salute San Raffaele. | Giada Piras nata 11 marzo 1969. lavorato Biesse. nata Macerata. conseguito titolo Ingegneria elettrica. lavora come commerciante. conseguito laurea Università Vita- Salute San Raffaele. | come | 37 | 22 | 24 | NaN |
| Gianni Chiesa è nato a Ravenna. Lavora per Leonardo. Ha scelto il percorso di Agraria. Lui ha visto la luce il 13 luglio 1998. Lavora come tipografo. Lui ha conseguito la laurea presso Università Telematica Internazionale UNINETTUNO. | Gianni Chiesa nato Ravenna. Lavora Leonardo. scelto percorso Agraria. visto luce 13 luglio 1998. Lavora tipografo. conseguito laurea Università Telematica Internazionale UNINETTUNO. | Gianni Chiesa nato Ravenna. Lavora Leonardo. scelto percorso Agraria. visto luce 13 luglio 1998. Lavora come tipografo. conseguito laurea Università Telematica Internazionale UNINETTUNO. | come | 37 | 22 | 23 | NaN |
| Gina Scalia lavora come osteopata. È nata il 12 marzo 1952. Ha i natali a Cervia. Si è laureata in Marketing. Ha fatto gli studi presso Università della Tuscia. Lei è impiegata presso Riso Scotti. | Gina Scalia lavora osteopata. nata 12 marzo 1952. Ha natali Cervia. Si laureata Marketing. fatto studi Università Tuscia. impiegata Riso Scotti. | Gina Scalia lavora come osteopata. nata 12 marzo 1952. natali Cervia. Si laureata Marketing. fatto studi Università Tuscia. impiegata Riso Scotti. | come | 35 | 21 | 21 | NaN |
| Giovanni Ghezzi ha fatto gli studi presso Università di Macerata. È originario di Trani. Esercita la professione di commercialista. Si è laureato in Cybersicurezza. Lui collabora con Parmalat. Ha come data di nascita il 9 marzo 1960. | Giovanni Ghezzi fatto studi Università Macerata. originario Trani. Esercita professione commercialista. Si laureato Cybersicurezza. collabora Parmalat. Ha data nascita 9 marzo 1960. | Giovanni Ghezzi fatto studi Università Macerata. originario Trani. Esercita professione commercialista. Si laureato Cybersicurezza. collabora Parmalat. come data nascita 9 marzo 1960. | come | 37 | 22 | 22 | NaN |
| Giulia Calabrese è nata a Civitanova Marche. Lei ha conseguito la laurea presso Università di Firenze Scuola Superiore. Lei esercita la professione di dietista. Collabora con Generali. Lei ha come data di nascita il 26 gennaio 1967. Lei ha seguito il corso di Ostetricia. | Giulia Calabrese nata Civitanova Marche. conseguito laurea Università Firenze Scuola Superiore. esercita professione dietista. Collabora Generali. ha data nascita 26 gennaio 1967. seguito corso Ostetricia. | Giulia Calabrese nata Civitanova Marche. conseguito laurea Università Firenze Scuola Superiore. esercita professione dietista. Collabora Generali. come data nascita 26 gennaio 1967. seguito corso Ostetricia. | come | 44 | 25 | 25 | NaN |
| Graziella Gigli esercita la professione di capitano. Lei si è laureata in Astronomia. Lei è originaria di Cerignola. Lei ha come data di nascita il 24 settembre 1989. Lei ha conseguito la laurea presso Università Europea di Roma. Lei collabora con Despar. | Graziella Gigli esercita professione capitano. si laureata Astronomia. originaria Cerignola. ha data nascita 24 settembre 1989. conseguito laurea Università Europea Roma. collabora Despar. | Graziella Gigli esercita professione capitano. si laureata Astronomia. originaria Cerignola. come data nascita 24 settembre 1989. conseguito laurea Università Europea Roma. collabora Despar. | come | 42 | 23 | 23 | NaN |
| Ilaria Castellano ha studiato presso Università di Verona. È originaria di Ruvo di Puglia. Ha conseguito il titolo in Agraria. Lavora come programmatore. Lei ha visto la luce il 16 gennaio 1957. È impiegata presso Saipem. | Ilaria Castellano studiato Università Verona. originaria Ruvo Puglia. conseguito titolo Agraria. Lavora programmatore. visto luce 16 gennaio 1957. impiegata Saipem. | Ilaria Castellano studiato Università Verona. originaria Ruvo Puglia. conseguito titolo Agraria. Lavora come programmatore. visto luce 16 gennaio 1957. impiegata Saipem. | come | 36 | 20 | 21 | NaN |
| Irene Carta lavora come scenografo. Lei ha visto la luce a Gorizia. Ha frequentato Università di Venezia. Lei collabora con Geox. Ha conseguito il titolo in Scienze della comunicazione digitale. Ha come data di nascita il 23 febbraio 1961. | Irene Carta lavora scenografo. visto luce Gorizia. frequentato Università Venezia. collabora Geox. conseguito titolo Scienze comunicazione digitale. Ha data nascita 23 febbraio 1961. | Irene Carta lavora come scenografo. visto luce Gorizia. frequentato Università Venezia. collabora Geox. conseguito titolo Scienze comunicazione digitale. come data nascita 23 febbraio 1961. | come | 39 | 23 | 24 | NaN |
| Irene Morandi ha lavorato presso Kappa. Lei ha studiato presso Università del Foro Italico. Si è laureata in Economia aziendale. Ha come data di nascita il 28 maggio 1971. Lei ha il ruolo di filosofo. È nata a Torre del Greco. | Irene Morandi lavorato Kappa. studiato Università Foro Italico. Si laureata Economia aziendale. Ha data nascita 28 maggio 1971. ha ruolo filosofo. nata Torre Greco. | Irene Morandi lavorato Kappa. studiato Università Foro Italico. Si laureata Economia aziendale. come data nascita 28 maggio 1971. ruolo filosofo. nata Torre Greco. | come | 41 | 24 | 23 | NaN |
| Isidoro Silvani ha frequentato Università di Pavia Collegio Universitario. Lui ha come data di nascita il 20 maggio 1960. È originario di Foligno. Lui collabora con De Longhi. Ha il ruolo di consulente. Ha scelto il percorso di Editoria. | Isidoro Silvani frequentato Università Pavia Collegio Universitario. ha data nascita 20 maggio 1960. originario Foligno. collabora De Longhi. Ha ruolo consulente. scelto percorso Editoria. | Isidoro Silvani frequentato Università Pavia Collegio Universitario. come data nascita 20 maggio 1960. originario Foligno. collabora De Longhi. ruolo consulente. scelto percorso Editoria. | come | 39 | 24 | 23 | NaN |
| Jolanda Greco collabora con Saras. Lei ha seguito il corso di Storia dell architettura. Lei lavora come musicista. Lei ha fatto gli studi presso Università di Catania Scuola Superiore. Lei è nata il 23 ottobre 1995. Lei ha i natali a La Spezia. | Jolanda Greco collabora Saras. seguito corso Storia architettura. lavora musicista. fatto studi Università Catania Scuola Superiore. nata 23 ottobre 1995. ha natali Spezia. | Jolanda Greco collabora Saras. seguito corso Storia dell architettura. lavora come musicista. fatto studi Università Catania Scuola Superiore. nata 23 ottobre 1995. natali Spezia. | come dell | 43 | 23 | 24 | NaN |
| Katia Pani ha studiato presso Università Magna Graecia di Catanzaro. È impiegata presso Bottega Veneta. Lei ha visto la luce il 9 aprile 1974. Lei lavora come attore. È originaria di Ravenna. Ha conseguito il titolo in Architettura. | Katia Pani studiato Università Magna Graecia Catanzaro. impiegata Bottega Veneta. visto luce 9 aprile 1974. lavora attore. originaria Ravenna. conseguito titolo Architettura. | Katia Pani studiato Università Magna Graecia Catanzaro. impiegata Bottega Veneta. visto luce 9 aprile 1974. lavora come attore. originaria Ravenna. conseguito titolo Architettura. | come | 38 | 22 | 23 | NaN |
| Laura Madau è originaria di Sondrio. Lei si è laureata in Ingegneria civile. Lei è impiegata presso Cucine Lube. È nata il 24 maggio 1984. Lavora come radiologo. Lei ha fatto gli studi presso Università di Brescia. | Laura Madau originaria Sondrio. si laureata Ingegneria civile. impiegata Cucine Lube. nata 24 maggio 1984. Lavora radiologo. fatto studi Università Brescia. | Laura Madau originaria Sondrio. si laureata Ingegneria civile. impiegata Cucine Lube. nata 24 maggio 1984. Lavora come radiologo. fatto studi Università Brescia. | come | 37 | 21 | 22 | NaN |
| Laura Tedesco ha scelto il percorso di Statistica. Ha conseguito la laurea presso Università di Bologna Scuola di Agraria. Lei è nata a Pordenone. Lei lavora per Campari. Lavora come dentista. È nata il 28 agosto 2002. | Laura Tedesco scelto percorso Statistica. conseguito laurea Università Bologna Scuola Agraria. nata Pordenone. lavora Campari. Lavora dentista. nata 28 agosto 2002. | Laura Tedesco scelto percorso Statistica. conseguito laurea Università Bologna Scuola Agraria. nata Pordenone. lavora Campari. Lavora come dentista. nata 28 agosto 2002. | come | 37 | 21 | 22 | NaN |
| Leonardo Pascale ha frequentato Università di Bologna Campus Cesena. Lui ha lavorato presso Italgas. Ha seguito il corso di Economia politica. Lui ha come data di nascita il 10 novembre 1983. Lavora come pescatore. Ha visto la luce a Vibo Valentia. | Leonardo Pascale frequentato Università Bologna Campus Cesena. lavorato Italgas. seguito corso Economia politica. ha data nascita 10 novembre 1983. Lavora pescatore. visto luce Vibo Valentia. | Leonardo Pascale frequentato Università Bologna Campus Cesena. lavorato Italgas. seguito corso Economia politica. come data nascita 10 novembre 1983. Lavora come pescatore. visto luce Vibo Valentia. | come | 41 | 25 | 26 | NaN |
| Letizia Vitiello lavora come fabbro. Ha come data di nascita il 4 novembre 1963. Lei lavora per Oto Melara. Lei ha conseguito il titolo in Conservazione dei beni culturali. Lei ha i natali a Latina. Lei ha frequentato Università di Teramo. | Letizia Vitiello lavora fabbro. Ha data nascita 4 novembre 1963. lavora Oto Melara. conseguito titolo Conservazione beni culturali. ha natali Latina. frequentato Università Teramo. | Letizia Vitiello lavora come fabbro. come data nascita 4 novembre 1963. lavora Oto Melara. conseguito titolo Conservazione beni culturali. natali Latina. frequentato Università Teramo. | come | 41 | 24 | 24 | NaN |
| Lidia Spada lavora per Benetton. Lei ha i natali a Lanciano. Lavora come psicoterapeuta. Ha frequentato Università di Palermo. Ha scelto il percorso di Bioinformatica. Lei ha come data di nascita il 15 novembre 2001. | Lidia Spada lavora Benetton. ha natali Lanciano. Lavora psicoterapeuta. frequentato Università Palermo. scelto percorso Bioinformatica. ha data nascita 15 novembre 2001. | Lidia Spada lavora Benetton. natali Lanciano. Lavora come psicoterapeuta. frequentato Università Palermo. scelto percorso Bioinformatica. come data nascita 15 novembre 2001. | come | 35 | 21 | 21 | NaN |
| Lidia Valentini è nata a Nocera Inferiore. Lei ha seguito il corso di Odontoiatria. Lei è nata il 3 aprile 1980. Ha conseguito la laurea presso Università Europea di Roma. Lei lavora come editore. Ha lavorato presso Candy. | Lidia Valentini nata Nocera Inferiore. seguito corso Odontoiatria. nata 3 aprile 1980. conseguito laurea Università Europea Roma. lavora editore. lavorato Candy. | Lidia Valentini nata Nocera Inferiore. seguito corso Odontoiatria. nata 3 aprile 1980. conseguito laurea Università Europea Roma. lavora come editore. lavorato Candy. | come | 38 | 21 | 22 | NaN |
| Loredana Vezzosi ha scelto il percorso di Architettura. Ha visto la luce il 24 aprile 1972. Lei lavora come scenografo. Ha conseguito la laurea presso Università del Salento. Lei è originaria di Gela. È impiegata presso Alfa Romeo. | Loredana Vezzosi scelto percorso Architettura. visto luce 24 aprile 1972. lavora scenografo. conseguito laurea Università Salento. originaria Gela. impiegata Alfa Romeo. | Loredana Vezzosi scelto percorso Architettura. visto luce 24 aprile 1972. lavora come scenografo. conseguito laurea Università Salento. originaria Gela. impiegata Alfa Romeo. | come | 38 | 21 | 22 | NaN |
| Lorena Marino ha come data di nascita il 9 marzo 1953. Ha studiato presso Università Europea di Roma. Lavora come falegname. Lei è nata a Genova. Lei collabora con Tenaris. Ha scelto il percorso di Astronomia. | Lorena Marino ha data nascita 9 marzo 1953. studiato Università Europea Roma. Lavora falegname. nata Genova. collabora Tenaris. scelto percorso Astronomia. | Lorena Marino come data nascita 9 marzo 1953. studiato Università Europea Roma. Lavora come falegname. nata Genova. collabora Tenaris. scelto percorso Astronomia. | come | 36 | 21 | 22 | NaN |
| Lorena Soldati ha conseguito la laurea presso Università di Catania Scuola Superiore. Ha conseguito il titolo in Medicina e chirurgia. Lei ha come data di nascita il 3 luglio 1970. Esercita la professione di guida turistica. Lei ha lavorato presso Danieli. Lei ha visto la luce a Grosseto. | Lorena Soldati conseguito laurea Università Catania Scuola Superiore. conseguito titolo Medicina e chirurgia. ha data nascita 3 luglio 1970. Esercita professione guida turistica. lavorato Danieli. visto luce Grosseto. | Lorena Soldati conseguito laurea Università Catania Scuola Superiore. conseguito titolo Medicina e chirurgia. come data nascita 3 luglio 1970. Esercita professione guida turistica. lavorato Danieli. visto luce Grosseto. | come | 48 | 28 | 28 | NaN |
| Luca Quaglia ha i natali a Giugliano. Lavora per Borsalino. Lui ha il ruolo di linguista. Ha studiato presso Università di Modena e Reggio Emilia. Lui ha scelto il percorso di Biotecnologie. Lui ha come data di nascita il 26 giugno 1972. | Luca Quaglia ha natali Giugliano. Lavora Borsalino. ha ruolo linguista. studiato Università Modena e Reggio Emilia. scelto percorso Biotecnologie. ha data nascita 26 giugno 1972. | Luca Quaglia natali Giugliano. Lavora Borsalino. ruolo linguista. studiato Università Modena e Reggio Emilia. scelto percorso Biotecnologie. come data nascita 26 giugno 1972. | come | 42 | 25 | 23 | NaN |
| Luisa Bettini ha come data di nascita il 8 febbraio 1950. Lei ha visto la luce a Forlì. Lei ha studiato presso Università di Milano Scuola di Giurisprudenza. Lavora come agente immobiliare. Lei si è laureata in Fisioterapia. Lavora per Saipem. | Luisa Bettini ha data nascita 8 febbraio 1950. visto luce Forlì. studiato Università Milano Scuola Giurisprudenza. Lavora agente immobiliare. si laureata Fisioterapia. Lavora Saipem. | Luisa Bettini come data nascita 8 febbraio 1950. visto luce Forlì. studiato Università Milano Scuola Giurisprudenza. Lavora come agente immobiliare. si laureata Fisioterapia. Lavora Saipem. | come | 41 | 24 | 25 | NaN |
| Maddalena Montanari ha seguito il corso di Cinema e audiovisivi. Ha fatto gli studi presso Università di Padova Scuola Galileiana. Ha come data di nascita il 10 agosto 1993. Lei ha il ruolo di psicologo. Lei è impiegata presso Edison. Lei ha visto la luce a Trapani. | Maddalena Montanari seguito corso Cinema e audiovisivi. fatto studi Università Padova Scuola Galileiana. Ha data nascita 10 agosto 1993. ha ruolo psicologo. impiegata Edison. visto luce Trapani. | Maddalena Montanari seguito corso Cinema e audiovisivi. fatto studi Università Padova Scuola Galileiana. come data nascita 10 agosto 1993. ruolo psicologo. impiegata Edison. visto luce Trapani. | come | 47 | 27 | 26 | NaN |
| Manuela Bruni ha visto la luce il 8 dicembre 1967. Lei ha conseguito la laurea presso Università di Padova Scuola di Medicina. Lei ha lavorato presso Avio. Lei lavora come falegname. Lei ha i natali a Benevento. Ha conseguito il titolo in Giornalismo. | Manuela Bruni visto luce 8 dicembre 1967. conseguito laurea Università Padova Scuola Medicina. lavorato Avio. lavora falegname. ha natali Benevento. conseguito titolo Giornalismo. | Manuela Bruni visto luce 8 dicembre 1967. conseguito laurea Università Padova Scuola Medicina. lavorato Avio. lavora come falegname. natali Benevento. conseguito titolo Giornalismo. | come | 43 | 23 | 23 | NaN |
| Manuela Villa è impiegata presso Unipol. È falegname. Lei ha frequentato Università della Tuscia. È originaria di Lucca. Lei ha visto la luce il 6 marzo 1973. Lei si è laureata in Ingegneria dell autoveicolo. | Manuela Villa impiegata Unipol. falegname. frequentato Università Tuscia. originaria Lucca. visto luce 6 marzo 1973. si laureata Ingegneria autoveicolo. | Manuela Villa impiegata Unipol. falegname. frequentato Università Tuscia. originaria Lucca. visto luce 6 marzo 1973. si laureata Ingegneria dell autoveicolo. | dell | 35 | 19 | 20 | NaN |
| Marisa Silvestri è nata il 17 marzo 1972. Lei collabora con Alfa Romeo. Lei lavora come libraio. Ha conseguito la laurea presso Scuola IMT Alti Studi Lucca. È originaria di Eboli. Lei si è laureata in Ingegneria edile. | Marisa Silvestri nata 17 marzo 1972. collabora Alfa Romeo. lavora libraio. conseguito laurea Scuola IMT Alti Studi Lucca. originaria Eboli. si laureata Ingegneria edile. | Marisa Silvestri nata 17 marzo 1972. collabora Alfa Romeo. lavora come libraio. conseguito laurea Scuola IMT Alti Studi Lucca. originaria Eboli. si laureata Ingegneria edile. | come | 38 | 24 | 25 | NaN |
| Matteo Pastore ha seguito il corso di Dietistica. Ha come data di nascita il 21 settembre 1999. Lavora come tassista. Ha lavorato presso Reply. Lui ha i natali a Sassari. Lui ha frequentato Università di Palermo Scuola di Medicina. | Matteo Pastore seguito corso Dietistica. Ha data nascita 21 settembre 1999. Lavora tassista. lavorato Reply. ha natali Sassari. frequentato Università Palermo Scuola Medicina. | Matteo Pastore seguito corso Dietistica. come data nascita 21 settembre 1999. Lavora come tassista. lavorato Reply. natali Sassari. frequentato Università Palermo Scuola Medicina. | come | 39 | 23 | 23 | NaN |
| Mauro Congiu ha fatto gli studi presso Università di Urbino Scuola di Scienze. Ha visto la luce a Pescara. Lui ha lavorato presso Missoni. Ha conseguito il titolo in Giornalismo. Lui lavora come meccanico. Ha come data di nascita il 13 dicembre 1993. | Mauro Congiu fatto studi Università Urbino Scuola Scienze. visto luce Pescara. lavorato Missoni. conseguito titolo Giornalismo. lavora meccanico. Ha data nascita 13 dicembre 1993. | Mauro Congiu fatto studi Università Urbino Scuola Scienze. visto luce Pescara. lavorato Missoni. conseguito titolo Giornalismo. lavora come meccanico. come data nascita 13 dicembre 1993. | come | 43 | 24 | 25 | NaN |
| Mauro Vitale ha lavorato presso Despar. Lui ha fatto gli studi presso Università del Piemonte Orientale. Lavora come meccanico. Ha scelto il percorso di Ingegneria chimica. Lui è nato a Campobasso. Lui è nato il 5 luglio 1997. | Mauro Vitale lavorato Despar. fatto studi Università Piemonte Orientale. Lavora meccanico. scelto percorso Ingegneria chimica. nato Campobasso. nato 5 luglio 1997. | Mauro Vitale lavorato Despar. fatto studi Università Piemonte Orientale. Lavora come meccanico. scelto percorso Ingegneria chimica. nato Campobasso. nato 5 luglio 1997. | come | 38 | 21 | 22 | NaN |
| Nadia Pastore ha conseguito la laurea presso Università Statale di Milano. È nata a Padova. Lei collabora con Carraro. Lei ha seguito il corso di Scienze naturali. Lei ha come data di nascita il 6 giugno 1979. È guida turistica. | Nadia Pastore conseguito laurea Università Statale Milano. nata Padova. collabora Carraro. seguito corso Scienze naturali. ha data nascita 6 giugno 1979. guida turistica. | Nadia Pastore conseguito laurea Università Statale Milano. nata Padova. collabora Carraro. seguito corso Scienze naturali. come data nascita 6 giugno 1979. guida turistica. | come | 40 | 23 | 23 | NaN |
| Nicola Miranda è nato il 4 maggio 2002. È originario di Lucca. Lui si è laureato in Sistemi informativi. Lui ha lavorato presso Hera. Lavora come tassista. Lui ha conseguito la laurea presso Università del Foro Italico. | Nicola Miranda nato 4 maggio 2002. originario Lucca. si laureato Sistemi informativi. lavorato Hera. Lavora tassista. conseguito laurea Università Foro Italico. | Nicola Miranda nato 4 maggio 2002. originario Lucca. si laureato Sistemi informativi. lavorato Hera. Lavora come tassista. conseguito laurea Università Foro Italico. | come | 37 | 21 | 22 | NaN |
| Nives Alberti ha come data di nascita il 19 agosto 1985. Ha il ruolo di chirurgo. Ha studiato presso Università del Sannio. Lei si è laureata in Ingegneria energetica. Ha visto la luce a Aprilia. È impiegata presso Brioni. | Nives Alberti ha data nascita 19 agosto 1985. Ha ruolo chirurgo. studiato Università Sannio. si laureata Ingegneria energetica. visto luce Aprilia. impiegata Brioni. | Nives Alberti come data nascita 19 agosto 1985. ruolo chirurgo. studiato Università Sannio. si laureata Ingegneria energetica. visto luce Aprilia. impiegata Brioni. | come | 39 | 23 | 22 | NaN |
| Orlando Franceschini ha seguito il corso di Ingegneria del territorio. Lui ha frequentato Università Link Campus. Lui ha come data di nascita il 22 gennaio 1998. Lui esercita la professione di dirigente. Collabora con Artemide. È nato a Siracusa. | Orlando Franceschini seguito corso Ingegneria territorio. frequentato Università Link Campus. ha data nascita 22 gennaio 1998. esercita professione dirigente. Collabora Artemide. nato Siracusa. | Orlando Franceschini seguito corso Ingegneria territorio. frequentato Università Link Campus. come data nascita 22 gennaio 1998. esercita professione dirigente. Collabora Artemide. nato Siracusa. | come | 39 | 23 | 23 | NaN |
| Ottavio Fiumara è dentista. Lui ha conseguito il titolo in Scienze motorie. Lui lavora per Carapelli. Ha fatto gli studi presso Scuola Universitaria Superiore IUSS Pavia. Ha come data di nascita il 26 agosto 1997. È nato a Benevento. | Ottavio Fiumara dentista. conseguito titolo Scienze motorie. lavora Carapelli. fatto studi Scuola Universitaria Superiore IUSS Pavia. Ha data nascita 26 agosto 1997. nato Benevento. | Ottavio Fiumara dentista. conseguito titolo Scienze motorie. lavora Carapelli. fatto studi Scuola Universitaria Superiore IUSS Pavia. come data nascita 26 agosto 1997. nato Benevento. | come | 39 | 24 | 24 | NaN |
| Pompeo Bellini esercita la professione di meccanico. Ha lavorato presso Segafredo. Lui è originario di Catania. Lui ha seguito il corso di Logopedia. Ha frequentato Università di Bergamo. Lui ha come data di nascita il 10 novembre 1986. | Pompeo Bellini esercita professione meccanico. lavorato Segafredo. originario Catania. seguito corso Logopedia. frequentato Università Bergamo. ha data nascita 10 novembre 1986. | Pompeo Bellini esercita professione meccanico. lavorato Segafredo. originario Catania. seguito corso Logopedia. frequentato Università Bergamo. come data nascita 10 novembre 1986. | come | 38 | 21 | 21 | NaN |
| Rebecca Pinna ha il ruolo di medico. Collabora con Fincantieri. Ha i natali a Forlì. Lei ha conseguito la laurea presso Università della Calabria. Ha come data di nascita il 7 settembre 1959. Lei ha scelto il percorso di Sociologia. | Rebecca Pinna ha ruolo medico. Collabora Fincantieri. Ha natali Forlì. conseguito laurea Università Calabria. Ha data nascita 7 settembre 1959. scelto percorso Sociologia. | Rebecca Pinna ruolo medico. Collabora Fincantieri. natali Forlì. conseguito laurea Università Calabria. come data nascita 7 settembre 1959. scelto percorso Sociologia. | come | 40 | 23 | 21 | NaN |
| Renato Piscopo è impiegato presso Vittoria Assicurazioni. Lui ha studiato presso Università di Camerino Scuola di Scienze. Ha seguito il corso di Urbanistica. Ha come data di nascita il 11 ottobre 1974. Lui ha i natali a Lodi. Lui esercita la professione di giornalista. | Renato Piscopo impiegato Vittoria Assicurazioni. studiato Università Camerino Scuola Scienze. seguito corso Urbanistica. Ha data nascita 11 ottobre 1974. ha natali Lodi. esercita professione giornalista. | Renato Piscopo impiegato Vittoria Assicurazioni. studiato Università Camerino Scuola Scienze. seguito corso Urbanistica. come data nascita 11 ottobre 1974. natali Lodi. esercita professione giornalista. | come | 44 | 25 | 24 | NaN |
| Rocco Cangemi lavora come professore. Lui collabora con Reale Mutua. Ha conseguito il titolo in Astronomia. Ha visto la luce a Foggia. Lui ha fatto gli studi presso Università di Milano Scuola di Giurisprudenza. Ha i natali il 29 aprile 1960. | Rocco Cangemi lavora professore. collabora Reale Mutua. conseguito titolo Astronomia. visto luce Foggia. fatto studi Università Milano Scuola Giurisprudenza. Ha natali 29 aprile 1960. | Rocco Cangemi lavora come professore. collabora Reale Mutua. conseguito titolo Astronomia. visto luce Foggia. fatto studi Università Milano Scuola Giurisprudenza. natali 29 aprile 1960. | come | 41 | 24 | 24 | NaN |
| Rosa Fanelli collabora con Dallara. Lei ha il ruolo di dentista. Lei ha conseguito il titolo in Chimica e tecnologia farmaceutiche. Lei è originaria di Pisa. Ha come data di nascita il 25 giugno 1996. Ha studiato presso Università Niccolò Cusano. | Rosa Fanelli collabora Dallara. ha ruolo dentista. conseguito titolo Chimica e tecnologia farmaceutiche. originaria Pisa. Ha data nascita 25 giugno 1996. studiato Università Niccolò Cusano. | Rosa Fanelli collabora Dallara. ruolo dentista. conseguito titolo Chimica e tecnologia farmaceutiche. originaria Pisa. come data nascita 25 giugno 1996. studiato Università Niccolò Cusano. | come | 41 | 25 | 24 | NaN |
| Rosalba Porta ha conseguito il titolo in Ingegneria biomedica. Lei ha frequentato Università di Bergamo. Collabora con Motta. È cuoco. Lei ha come data di nascita il 16 settembre 1962. Lei ha i natali a Sora. | Rosalba Porta conseguito titolo Ingegneria biomedica. frequentato Università Bergamo. Collabora Motta. cuoco. ha data nascita 16 settembre 1962. ha natali Sora. | Rosalba Porta conseguito titolo Ingegneria biomedica. frequentato Università Bergamo. Collabora Motta. cuoco. come data nascita 16 settembre 1962. natali Sora. | come | 36 | 21 | 20 | NaN |
| Sebastiano Gigli ha conseguito il titolo in Cinema e audiovisivi. Lui ha fatto gli studi presso Università di Napoli Suor Orsola. Lavora per Dedagroup. È originario di Bolzano. Lavora come orafo. Lui ha visto la luce il 22 febbraio 1967. | Sebastiano Gigli conseguito titolo Cinema e audiovisivi. fatto studi Università Napoli Suor Orsola. Lavora Dedagroup. originario Bolzano. Lavora orafo. visto luce 22 febbraio 1967. | Sebastiano Gigli conseguito titolo Cinema e audiovisivi. fatto studi Università Napoli Suor Orsola. Lavora Dedagroup. originario Bolzano. Lavora come orafo. visto luce 22 febbraio 1967. | come | 40 | 24 | 25 | NaN |
| Serena Lombardi ha studiato presso Università Bocconi. È nata a Monza. Ha il ruolo di pescatore. È impiegata presso Galbani. Ha come data di nascita il 11 settembre 1996. Ha scelto il percorso di Fisica. | Serena Lombardi studiato Università Bocconi. nata Monza. Ha ruolo pescatore. impiegata Galbani. Ha data nascita 11 settembre 1996. scelto percorso Fisica. | Serena Lombardi studiato Università Bocconi. nata Monza. ruolo pescatore. impiegata Galbani. come data nascita 11 settembre 1996. scelto percorso Fisica. | come | 35 | 21 | 20 | NaN |
| Stefania Mattei ha seguito il corso di Conservazione dei beni culturali. Lei è nata a Palermo. Lavora come insegnante. Lei è impiegata presso Candy. Lei è nata il 29 maggio 1990. Lei ha conseguito la laurea presso Università di Parma Scuola di Medicina. | Stefania Mattei seguito corso Conservazione beni culturali. nata Palermo. Lavora insegnante. impiegata Candy. nata 29 maggio 1990. conseguito laurea Università Parma Scuola Medicina. | Stefania Mattei seguito corso Conservazione beni culturali. nata Palermo. Lavora come insegnante. impiegata Candy. nata 29 maggio 1990. conseguito laurea Università Parma Scuola Medicina. | come | 43 | 23 | 24 | NaN |
| Stella Prisco ha seguito il corso di Astronomia. Ha lavorato presso Colmar. Lavora come catechista. Lei ha conseguito la laurea presso Università di Messina. Ha i natali a Pisa. Lei è nata il 16 luglio 1980. | Stella Prisco seguito corso Astronomia. lavorato Colmar. Lavora catechista. conseguito laurea Università Messina. Ha natali Pisa. nata 16 luglio 1980. | Stella Prisco seguito corso Astronomia. lavorato Colmar. Lavora come catechista. conseguito laurea Università Messina. natali Pisa. nata 16 luglio 1980. | come | 36 | 20 | 20 | NaN |
| Tina Zito ha frequentato Università Pontificia Salesiana. Lei ha il ruolo di tornitore. Ha come data di nascita il 3 novembre 1975. Lei ha seguito il corso di Scienze infermieristiche. Collabora con Alessi. Lei è originaria di Martina Franca. | Tina Zito frequentato Università Pontificia Salesiana. ha ruolo tornitore. Ha data nascita 3 novembre 1975. seguito corso Scienze infermieristiche. Collabora Alessi. originaria Martina Franca. | Tina Zito frequentato Università Pontificia Salesiana. ruolo tornitore. come data nascita 3 novembre 1975. seguito corso Scienze infermieristiche. Collabora Alessi. originaria Martina Franca. | come | 39 | 24 | 23 | NaN |
| Tiziana Usai ha come data di nascita il 3 dicembre 2000. Ha frequentato Università Cattolica del Sacro Cuore. È paesaggista. Lei è impiegata presso Exprivia. Lei è originaria di Padova. Ha seguito il corso di Scienze agrarie. | Tiziana Usai ha data nascita 3 dicembre 2000. frequentato Università Cattolica Sacro Cuore. paesaggista. impiegata Exprivia. originaria Padova. seguito corso Scienze agrarie. | Tiziana Usai come data nascita 3 dicembre 2000. frequentato Università Cattolica Sacro Cuore. paesaggista. impiegata Exprivia. originaria Padova. seguito corso Scienze agrarie. | come | 37 | 22 | 22 | NaN |
| Tullio Napolitano è nato il 25 giugno 1963. Ha i natali a Carpi. Lavora per Peroni. Ha conseguito il titolo in Veterinaria. Lavora come postino. Lui ha conseguito la laurea presso Università di Padova. | Tullio Napolitano nato 25 giugno 1963. Ha natali Carpi. Lavora Peroni. conseguito titolo Veterinaria. Lavora postino. conseguito laurea Università Padova. | Tullio Napolitano nato 25 giugno 1963. natali Carpi. Lavora Peroni. conseguito titolo Veterinaria. Lavora come postino. conseguito laurea Università Padova. | come | 34 | 20 | 20 | NaN |
| Ulisse Carta ha frequentato Università di Roma Foro Italico. Collabora con Bialetti. È originario di Jesolo. Lui ha come data di nascita il 1 gennaio 1980. Ha conseguito il titolo in Cinema e audiovisivi. Lui è infermiere. | Ulisse Carta frequentato Università Roma Foro Italico. Collabora Bialetti. originario Jesolo. ha data nascita 1 gennaio 1980. conseguito titolo Cinema e audiovisivi. infermiere. | Ulisse Carta frequentato Università Roma Foro Italico. Collabora Bialetti. originario Jesolo. come data nascita 1 gennaio 1980. conseguito titolo Cinema e audiovisivi. infermiere. | come | 37 | 23 | 23 | NaN |
| Valeria Montella lavora come economista. Ha conseguito il titolo in Editoria. Lei ha come data di nascita il 13 gennaio 1976. È originaria di Napoli. Lei ha fatto gli studi presso Università Europea di Roma. Lei collabora con De Longhi. | Valeria Montella lavora economista. conseguito titolo Editoria. ha data nascita 13 gennaio 1976. originaria Napoli. fatto studi Università Europea Roma. collabora De Longhi. | Valeria Montella lavora come economista. conseguito titolo Editoria. come data nascita 13 gennaio 1976. originaria Napoli. fatto studi Università Europea Roma. collabora De Longhi. | come | 40 | 23 | 24 | NaN |
| Valerio Vaccaro ha seguito il corso di Chimica e tecnologia farmaceutiche. Lavora per Max Mara. Ha frequentato Università di Torino. Ha visto la luce a Formia. Lui lavora come capitano. Lui ha come data di nascita il 9 gennaio 1983. | Valerio Vaccaro seguito corso Chimica e tecnologia farmaceutiche. Lavora Max Mara. frequentato Università Torino. visto luce Formia. lavora capitano. ha data nascita 9 gennaio 1983. | Valerio Vaccaro seguito corso Chimica e tecnologia farmaceutiche. Lavora Max Mara. frequentato Università Torino. visto luce Formia. lavora come capitano. come data nascita 9 gennaio 1983. | come | 40 | 25 | 26 | NaN |
| Vincenzo Grassi collabora con Magneti Marelli. Lui ha conseguito la laurea presso Università di Milano Scuola di Giurisprudenza. Lui ha visto la luce a Frosinone. Lavora come bibliotecario. Ha visto la luce il 13 luglio 1967. Lui ha conseguito il titolo in Ingegneria chimica industriale. | Vincenzo Grassi collabora Magneti Marelli. conseguito laurea Università Milano Scuola Giurisprudenza. visto luce Frosinone. Lavora bibliotecario. visto luce 13 luglio 1967. conseguito titolo Ingegneria chimica industriale. | Vincenzo Grassi collabora Magneti Marelli. conseguito laurea Università Milano Scuola Giurisprudenza. visto luce Frosinone. Lavora come bibliotecario. visto luce 13 luglio 1967. conseguito titolo Ingegneria chimica industriale. | come | 45 | 26 | 27 | NaN |
| Viola Bruni ha i natali il 27 ottobre 1975. Lei ha seguito il corso di Scienze politiche. Ha i natali a Ostuni. Lei lavora come orafo. Lei collabora con Salvatore Ferragamo. Ha frequentato Università Pontificia Salesiana. | Viola Bruni ha natali 27 ottobre 1975. seguito corso Scienze politiche. Ha natali Ostuni. lavora orafo. collabora Salvatore Ferragamo. frequentato Università Pontificia Salesiana. | Viola Bruni natali 27 ottobre 1975. seguito corso Scienze politiche. natali Ostuni. lavora come orafo. collabora Salvatore Ferragamo. frequentato Università Pontificia Salesiana. | come | 36 | 23 | 22 | NaN |
| Zaira Castaldi ha fatto gli studi presso Università di Pavia Collegio Universitario. Ha visto la luce a Ferrara. Ha come data di nascita il 16 gennaio 1970. Lavora per Rizzoli. Si è laureata in Ingegneria dei materiali. Ha il ruolo di programmatore. | Zaira Castaldi fatto studi Università Pavia Collegio Universitario. visto luce Ferrara. Ha data nascita 16 gennaio 1970. Lavora Rizzoli. Si laureata Ingegneria materiali. Ha ruolo programmatore. | Zaira Castaldi fatto studi Università Pavia Collegio Universitario. visto luce Ferrara. come data nascita 16 gennaio 1970. Lavora Rizzoli. Si laureata Ingegneria materiali. ruolo programmatore. | come | 42 | 26 | 25 | NaN |

## E: pipeline

| format | size | batch | n_questions | prompt_tokens_mean | answer_tokens_mean | strip_cpu_ms | core_gpu_ms | realize_cpu_ms | total_ms | pieno_ms | ratio_vs_pieno | core_flops | pieno_flops | core_acc | e2e_acc | pieno_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| telegrafico_auto | 0.60M | 1 | 1000 | 4.658 | 2.844 | 1.345 | 3.77 | 0.03502 | 5.15 | 3.55 | 1.451 | 1.903e+07 | 2.305e+07 | 0.921 | 0.921 | 0.933 |
| telegrafico_auto | 0.60M | 64 | 1000 | 4.658 | 2.844 | 69.87 | 8.478 | 1.287 | 79.64 | 6.268 | 12.71 | 2.753e+07 | 2.937e+07 | 0.921 | 0.921 | 0.933 |
| telegrafico_auto | 1.49M | 1 | 1000 | 4.658 | 2.844 | 1.333 | 3.7 | 0.02462 | 5.058 | 3.498 | 1.446 | 3.964e+07 | 4.811e+07 | 0.996 | 0.997 | 1 |
| telegrafico_auto | 1.49M | 64 | 1000 | 4.658 | 2.844 | 70.17 | 8.713 | 0.6296 | 79.51 | 9.007 | 8.828 | 5.29e+07 | 6.129e+07 | 0.996 | 0.997 | 1 |
| telegrafico_auto | 4M | 1 | 1000 | 4.658 | 2.844 | 1.343 | 3.971 | 0.02483 | 5.339 | 3.641 | 1.466 | 9.273e+07 | 1.128e+08 | 0.996 | 0.997 | 1 |
| telegrafico_auto | 4M | 64 | 1000 | 4.658 | 2.844 | 69.59 | 10.33 | 0.6063 | 80.53 | 12.03 | 6.694 | 1.237e+08 | 1.436e+08 | 0.996 | 0.997 | 1 |
| telegrafico_lista | 0.60M | 1 | 1000 | 4.667 | 2.848 | 0.01087 | 3.233 | 0.02793 | 3.272 | 3.278 | 0.9983 | 1.881e+07 | 2.305e+07 | 0.978 | 0.98 | 0.933 |
| telegrafico_lista | 0.60M | 64 | 1000 | 4.667 | 2.848 | 0.2283 | 5.518 | 0.8762 | 6.622 | 7.005 | 0.9453 | 2.511e+07 | 2.937e+07 | 0.978 | 0.98 | 0.933 |
| telegrafico_lista | 1.49M | 1 | 1000 | 4.667 | 2.848 | 0.01084 | 3.274 | 0.02339 | 3.308 | 3.426 | 0.9655 | 3.928e+07 | 4.811e+07 | 0.998 | 1 | 1 |
| telegrafico_lista | 1.49M | 64 | 1000 | 4.667 | 2.848 | 0.2287 | 6.791 | 0.6 | 7.62 | 6.613 | 1.152 | 4.804e+07 | 6.129e+07 | 0.998 | 1 | 1 |
| telegrafico_lista | 4M | 1 | 1000 | 4.667 | 2.848 | 0.01072 | 3.379 | 0.02321 | 3.413 | 3.543 | 0.9632 | 9.215e+07 | 1.128e+08 | 0.998 | 1 | 1 |
| telegrafico_lista | 4M | 64 | 1000 | 4.667 | 2.848 | 0.2487 | 8.411 | 0.6371 | 9.297 | 10.98 | 0.8467 | 1.127e+08 | 1.436e+08 | 0.998 | 1 | 1 |
| interlingua | 0.60M | 1 | 1000 | 5.668 | 4.251 | 0.003933 | 2.256 | 0.03201 | 2.292 | 3.354 | 0.6834 | 2.094e+07 | 2.305e+07 | 0.935 | 0.935 | 0.933 |
| interlingua | 0.60M | 64 | 1000 | 5.668 | 4.251 | 0.08534 | 6.954 | 0.1907 | 7.23 | 7.281 | 0.993 | 3.57e+07 | 2.937e+07 | 0.936 | 0.936 | 0.933 |
| interlingua | 1.49M | 1 | 1000 | 5.668 | 4.251 | 0.003951 | 2.396 | 0.01351 | 2.413 | 3.302 | 0.7308 | 4.372e+07 | 4.811e+07 | 1 | 1 | 1 |
| interlingua | 1.49M | 64 | 1000 | 5.668 | 4.251 | 0.08626 | 6.493 | 0.1897 | 6.769 | 9.238 | 0.7327 | 7.449e+07 | 6.129e+07 | 1 | 1 | 1 |
| interlingua | 4M | 1 | 1000 | 5.668 | 4.251 | 0.003981 | 2.445 | 0.01347 | 2.462 | 3.639 | 0.6766 | 1.025e+08 | 1.128e+08 | 1 | 1 | 1 |
| interlingua | 4M | 64 | 1000 | 5.668 | 4.251 | 0.08641 | 7.202 | 0.1933 | 7.481 | 8.359 | 0.895 | 1.745e+08 | 1.436e+08 | 1 | 1 | 1 |
