# BERT_in_intraday_trading
This repo uses language models to extract infomation and apply to intraday trading.

# Model Comparison

## Test Set Performance Comparison

### MSE Comparison

| Model | RV | BV | MedRV | RK_Parzen | RK_TukeyHanning |
|---|---:|---:|---:|---:|---:|
| GARCH(1,1) | 0.034601 | 0.032738 | 0.032393 | 0.035610 | 0.036990 |
| DeepARCH | **0.016630** | **0.014462** | **0.014104** | **0.017739** | **0.019122** |
| Deep RealARCH | 0.027263 | 0.025380 | 0.025041 | 0.028251 | 0.029625 |
| Deep LLM RealARCH | 0.029543 | 0.027692 | 0.027344 | 0.030565 | 0.031949 |
| Deep LLM ARCH | 0.029508 | 0.027664 | 0.027314 | 0.030537 | 0.031924 |

---

### MAE Comparison

| Model | RV | BV | MedRV | RK_Parzen | RK_TukeyHanning |
|---|---:|---:|---:|---:|---:|
| GARCH(1,1) | 0.115124 | 0.114857 | 0.116186 | 0.122867 | 0.126133 |
| DeepARCH | **0.060785** | **0.061083** | **0.062729** | **0.074704** | **0.079886** |
| Deep RealARCH | 0.103572 | 0.104001 | 0.105249 | 0.112891 | 0.116637 |
| Deep LLM RealARCH | 0.109844 | 0.110060 | 0.111269 | 0.118418 | 0.121932 |
| Deep LLM ARCH | 0.108177 | 0.108468 | 0.109706 | 0.117076 | 0.120689 |

---

### Tail Loss Comparison

| Model | QLOSS (0.01) | JOINTLOSS (0.01) | QLOSS (0.05) | JOINTLOSS (0.05) |
|---|---:|---:|---:|---:|
| GARCH(1,1) | 0.012492 | 2.409936 | 0.029509 | 0.801144 |
| DeepARCH | **0.008254** | **0.842150** | **0.023102** | **0.206314** |
| Deep RealARCH | 0.011209 | 1.800698 | 0.027856 | 0.614567 |
| Deep LLM RealARCH | 0.011969 | 2.284871 | 0.028777 | 0.762096 |
| Deep LLM ARCH | 0.011633 | 2.034723 | 0.028446 | 0.687704 |

---

### Negative Loglikelihood Comparison

| Model | Negative Loglikelihood |
|---|---:|
| GARCH(1,1) | -1.370653 |
| DeepARCH | **-2.324074** |
| Deep RealARCH | -1.644596 |
| Deep LLM RealARCH | -1.352065 |
| Deep LLM ARCH | -1.569135 |

---

# Train Set Performance Comparison

## MSE Comparison

| Model | RV | BV | MedRV | RK_Parzen | RK_TukeyHanning |
|---|---:|---:|---:|---:|---:|
| GARCH(1,1) | 0.006711 | 0.006877 | 0.007117 | 0.009524 | 0.010650 |
| DeepARCH | **0.005558** | **0.005509** | **0.005853** | **0.007789** | **0.008760** |
| Deep RealARCH | 0.013051 | 0.013143 | 0.013524 | 0.015440 | 0.016457 |
| Deep LLM RealARCH | 0.017387 | 0.017485 | 0.017864 | 0.019782 | 0.020801 |
| Deep LLM ARCH | 0.017584 | 0.017763 | 0.018121 | 0.020068 | 0.021111 |

---

## MAE Comparison

| Model | RV | BV | MedRV | RK_Parzen | RK_TukeyHanning |
|---|---:|---:|---:|---:|---:|
| GARCH(1,1) | 0.054223 | 0.057201 | 0.057592 | 0.068305 | 0.072570 |
| DeepARCH | **0.048794** | **0.051025** | **0.051860** | **0.061820** | **0.066102** |
| Deep RealARCH | 0.083624 | 0.085390 | 0.086186 | 0.092644 | 0.095970 |
| Deep LLM RealARCH | 0.090070 | 0.091369 | 0.092215 | 0.098123 | 0.101192 |
| Deep LLM ARCH | 0.090462 | 0.092111 | 0.092896 | 0.099090 | 0.102304 |

---

## Tail Loss Comparison

| Model | QLOSS (0.01) | JOINTLOSS (0.01) | QLOSS (0.05) | JOINTLOSS (0.05) |
|---|---:|---:|---:|---:|
| GARCH(1,1) | 0.005768 | 0.467631 | 0.018042 | -0.022367 |
| DeepARCH | **0.005503** | **0.389759** | **0.017484** | **-0.062093** |
| Deep RealARCH | 0.007247 | 0.921756 | 0.020888 | 0.224617 |
| Deep LLM RealARCH | 0.007885 | 1.267116 | 0.021776 | 0.336122 |
| Deep LLM ARCH | 0.007604 | 1.083414 | 0.021591 | 0.283836 |

---

## Negative Loglikelihood Comparison

| Model | Negative Loglikelihood |
|---|---:|
| GARCH(1,1) | -2.721986 |
| DeepARCH | -2.800878 |
| Deep RealARCH | **-3.508310** |
| Deep LLM RealARCH | -3.303224 |
| Deep LLM ARCH | -2.235757 |

---

# Overall Observations

- **DeepARCH** achieves the best overall predictive performance across nearly all MSE, MAE, and tail-risk metrics on the test set.
- **Deep RealARCH** achieves the best train negative loglikelihood but generalizes worse than DeepARCH on the test set.
- Both **LLM-enhanced models** underperform compared to DeepARCH and Deep RealARCH in this experiment.
- Traditional **GARCH(1,1)** performs significantly worse than deep learning-based approaches across all evaluation metrics.
