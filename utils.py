import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def evaluate(y_true, y_pred):
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc = clustering_accuracy(y_true, y_pred)
    return nmi, ari, acc
```

4. "Add evaluation utilities"
5. Commit

---

## قدم 7: تنظیمات نهایی

**اضافه کردن Topics:**
1. بالای صفحه سمت راست، یک قسمت About هست با یک آیکون چرخ دنده
2. کلیک کن روش
3. تو قسمت Topics بنویس (هر کدوم رو با Enter جدا کن):
   - `graph-neural-networks`
   - `deep-learning`
   - `pytorch`
   - `graph-clustering`

4. Save changes

---

## تموم شد!

**الان repository تو آماده است:**
- لینکش: `https://github.com/YOUR_USERNAME/DAH-GCN`

**تو مقاله بنویس:**
```
Code is available at: https://github.com/YOUR_USERNAME/DAH-GCN
