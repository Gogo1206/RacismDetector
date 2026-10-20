# Test results

| Model | runs | pr_auc | roc_auc | f1 | precision | recall | fpr | fpr_race_supportive |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 1 | 0.394 | 0.867 | 0.404 | 0.332 | 0.517 | 0.074 | 0.111 |
| DistilRoBERTa single-task | 3 | 0.475 ± 0.016 | 0.907 ± 0.009 | 0.470 ± 0.002 | 0.426 ± 0.041 | 0.535 ± 0.073 | 0.052 ± 0.016 | 0.068 ± 0.021 |
| DistilRoBERTa multi-task (λ=1) | 3 | 0.495 ± 0.007 | 0.913 ± 0.005 | 0.498 ± 0.016 | 0.458 ± 0.028 | 0.552 ± 0.072 | 0.047 ± 0.011 | 0.048 ± 0.011 |
