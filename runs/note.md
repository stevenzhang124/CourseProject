> source=labeled, target=unlabeled

|Method| Acc |
| ---|---|
| FL, target=non-iid| 58.6 |
| FL, target=iid | 58.6 |
| FL, only train on source (lamda = 0) | 58.2 |
| baseline1 = No FL; only train on source | 52.0 |
| baseline2 = no FL;  train on source + target domain (non-iid)| 53.3 |
