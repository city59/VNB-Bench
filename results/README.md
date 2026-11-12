# Additional results

All numbers are on the test period. Conversion and re-engagement are ccNDCG@10, conditional conversion is AUC,
and brackets give 95% user-bootstrap intervals (1,000 resamples).

## 1. Standard protocol: where models rank a held-out purchase already in the user's history

The standard protocols rank each user's first test-period purchase against 99 negatives outside the user's history
(Sampled) or against the full catalog with earlier purchases masked (Full). The table covers test users whose
held-out purchase already appears among their earlier events. Columns give the share ranked first under Sampled,
and the median rank and the top-10 share under Full.

| Log (items) | Model | First among 100 sampled | Median full-catalog rank | Full-catalog top 10 |
|---|---|---|---|---|
| Tmall (2.39M) | MB-STR | 45% | 22,553 | 3% |
| Tmall | PBAT | 36% | 53,305 | 4% |
| Tmall | MISSL | 58% | 4,661 | 7% |
| Tmall | MBSRec | 83% | 18 | 45% |
| Tmall | MBGen | 99% | 1 | 97% |
| RetailRocket (193K) | MB-STR | 83% | 10 | 51% |
| RetailRocket | PBAT | 83% | 16 | 46% |
| RetailRocket | MISSL | 91% | 2 | 78% |

## 2. Fine-tuning on conversion labels

"Official objective" is the model of the main results. Fine-tuning starts from a retrained checkpoint with the same
configuration and uses a listwise softmax over the candidates of training-period
queries (all converted candidates and up to 49 others per query, Adam, learning rate 1e-4, at most 30 epochs,
early stopping with patience 3 on validation conversion). On IJCAI-15, fine-tuning never improved MBSRec's
validation score, so its starting checkpoint is kept.

| Log | Scorer | Conversion | Re-engagement | Conditional AUC |
|---|---|---|---|---|
| Tmall | MBSRec, official objective | 0.394 [0.371, 0.417] | 0.226 [0.219, 0.232] | 0.602 [0.580, 0.621] |
| Tmall | MBSRec, fine-tuned | 0.490 [0.469, 0.511] | 0.280 [0.274, 0.286] | 0.666 [0.646, 0.685] |
| Tmall | MBGen, official objective | 0.609 [0.589, 0.628] | 0.348 [0.342, 0.354] | 0.708 [0.687, 0.726] |
| Tmall | MBGen, fine-tuned | 0.629 [0.612, 0.648] | 0.368 [0.361, 0.375] | 0.717 [0.698, 0.738] |
| Tmall | Cart-first | 0.670 [0.653, 0.688] | 0.467 [0.461, 0.474] | 0.711 [0.690, 0.730] |
| Tmall | GBDT-UAI | 0.715 [0.699, 0.731] | 0.473 [0.467, 0.479] | 0.741 [0.722, 0.760] |
| IJCAI-15 | MBSRec, official objective | 0.215 [0.199, 0.230] | 0.162 [0.155, 0.169] | 0.540 [0.523, 0.557] |
| IJCAI-15 | MBSRec, fine-tuned | 0.215 [0.199, 0.230] | 0.162 [0.155, 0.169] | 0.540 [0.523, 0.557] |
| IJCAI-15 | MBGen, official objective | 0.239 [0.223, 0.256] | 0.189 [0.182, 0.196] | 0.546 [0.530, 0.563] |
| IJCAI-15 | MBGen, fine-tuned | 0.258 [0.241, 0.274] | 0.217 [0.211, 0.224] | 0.569 [0.553, 0.587] |
| IJCAI-15 | Cart-first | 0.282 [0.270, 0.294] | 0.286 [0.281, 0.292] | 0.539 [0.526, 0.552] |
| IJCAI-15 | GBDT-UAI | 0.374 [0.359, 0.390] | 0.305 [0.298, 0.311] | 0.596 [0.580, 0.613] |

## 3. Static-graph models with a frozen training-period graph

MEMBER and CHARM are trained on per-behavior user-item graphs built from the training period only, with their
official configurations, and the graph is not updated afterwards. Users without training-period events have no
edges, so every scorer is evaluated on the test queries of users seen in training. NoVa has no public code.
RetailRocket is omitted because only 373 of its 2,322 test users appear in the training period.

| Log | Test queries kept | Scorer | Conversion | Re-engagement | Conditional AUC |
|---|---|---|---|---|---|
| Tmall | 99% | MEMBER | 0.039 [0.017, 0.060] | 0.044 [0.037, 0.050] | 0.501 [0.478, 0.526] |
| Tmall | 99% | CHARM | 0.016 [-0.005, 0.036] | -0.002 [-0.009, 0.005] | 0.527 [0.502, 0.550] |
| Tmall | 99% | Popularity | 0.009 [-0.011, 0.029] | 0.004 [-0.003, 0.010] | 0.489 [0.469, 0.510] |
| Tmall | 99% | Recency | 0.445 [0.424, 0.465] | 0.255 [0.248, 0.261] | 0.648 [0.625, 0.670] |
| Tmall | 99% | Cart-first | 0.670 [0.652, 0.687] | 0.468 [0.462, 0.474] | 0.711 [0.691, 0.731] |
| Tmall | 99% | GBDT-UAI | 0.715 [0.698, 0.732] | 0.473 [0.467, 0.480] | 0.741 [0.722, 0.760] |
| Tmall | 99% | MBGen | 0.609 [0.589, 0.626] | 0.348 [0.342, 0.354] | 0.707 [0.686, 0.725] |
| IJCAI-15 | 95% | MEMBER | 0.030 [0.014, 0.046] | 0.043 [0.036, 0.050] | 0.502 [0.484, 0.520] |
| IJCAI-15 | 95% | CHARM | 0.119 [0.102, 0.136] | 0.046 [0.039, 0.053] | 0.546 [0.528, 0.563] |
| IJCAI-15 | 95% | Popularity | 0.123 [0.108, 0.139] | 0.059 [0.052, 0.066] | 0.561 [0.544, 0.578] |
| IJCAI-15 | 95% | Recency | 0.216 [0.207, 0.225] | 0.141 [0.136, 0.146] | 0.558 [0.548, 0.567] |
| IJCAI-15 | 95% | Cart-first | 0.282 [0.267, 0.294] | 0.288 [0.282, 0.294] | 0.539 [0.525, 0.552] |
| IJCAI-15 | 95% | GBDT-UAI | 0.376 [0.360, 0.391] | 0.306 [0.299, 0.313] | 0.596 [0.579, 0.613] |
| IJCAI-15 | 95% | MBGen | 0.245 [0.229, 0.262] | 0.189 [0.182, 0.196] | 0.548 [0.530, 0.567] |
| Taobao | 99% | MEMBER | 0.060 [0.048, 0.073] | 0.097 [0.093, 0.101] | 0.502 [0.488, 0.516] |
| Taobao | 99% | CHARM | 0.093 [0.081, 0.106] | 0.025 [0.021, 0.029] | 0.552 [0.536, 0.566] |
| Taobao | 99% | Popularity | 0.100 [0.089, 0.111] | 0.031 [0.027, 0.035] | 0.559 [0.546, 0.572] |
| Taobao | 99% | Recency | 0.165 [0.152, 0.178] | 0.093 [0.088, 0.097] | 0.568 [0.554, 0.582] |
| Taobao | 99% | Cart-first | 0.523 [0.511, 0.535] | 0.382 [0.378, 0.387] | 0.659 [0.646, 0.672] |
| Taobao | 99% | GBDT-UAI | 0.552 [0.539, 0.563] | 0.392 [0.388, 0.397] | 0.679 [0.665, 0.692] |
| Taobao | 99% | MBGen | 0.285 [0.273, 0.297] | 0.202 [0.198, 0.206] | 0.605 [0.591, 0.619] |
