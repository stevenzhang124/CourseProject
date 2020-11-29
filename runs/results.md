> source=labeled, target=unlabeled

<br>

## Methods (see plot in tensorboard)
1. FL + non-iid + ddm (lam=5): local_ep=1  "07_20-06-40"
1. FL + non-iid + ddm (lam=5): local_ep=5   "07_20-06-51"
1. FL + non-iid + no_ddm (lam=0): local_ep=1    "07_20-27-52"
1. no FL + non-iid + ddm (baseline_2, lam=5)    "28_17-01-42"
1. no FL + non-iid + no_ddm (baseline_1)    "28_17-06-23"
1. no grl
1. with grl
<br>

## Results: Smooth rate 0.95
| Method | Acc |
|---|---|
| 1 | 55.2 |
| 2 | 53.8 |
| 3 | 53.0 |
| 4 | 55.7 |
| 5 | 53.7 |
