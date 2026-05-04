# Model Evaluation Metrics

This document explains every metric reported when running the evaluation:

```bash
python scripts/04_forecast.py -f weekly evaluate
python scripts/04_forecast.py evaluate --all
```

---

## Global Metrics

These metrics are computed across all item-period pairs in the test set (last 12 weeks for weekly, last 12 weeks for daily).

### Global R2 (R-squared)

**What it measures**: Proportion of variance in actual sales explained by the model.

- Range: 0 to 1 (higher is better)
- R2 = 1.0 means perfect prediction
- R2 = 0.0 means the model is no better than predicting the average
- R2 < 0 means the model is worse than the mean

**Interpretation for this project**: A weekly R2 of ~0.83 means the model captures most seasonal and trend patterns, but some demand fluctuations remain unpredictable.

**Limitation**: R2 is sensitive to outliers and can be inflated by predicting high-volume items well while ignoring slow movers. It is a holistic measure but not sufficient alone.

---

### Global wMAPE (Weighted Mean Absolute Percentage Error)

**What it measures**: Average percentage error, weighted by actual volume.

```
wMAPE = (|actual - predicted| summed over all rows) / (actual summed over all rows) * 100
```

- Range: 0% to infinity (lower is better)
- 0% means perfect prediction
- Each row contributes equally to the error pool, then divided by total actual volume

**Interpretation for this project**: A wMAPE of ~21% means that, on average, the model's predictions deviate by about 21% of actual sales volume. High-volume items (Kopi Susu, Lychee Tea) dominate this metric.

**Why this is the most useful single metric**: It directly translates to business impact. A 21% wMAPE means you should expect to over-order or under-order by roughly 1/5th of actual demand on average. It penalizes large absolute errors on popular items more than small errors on slow movers, which aligns with supply planning priorities.

---

### Global MAE (Mean Absolute Error)

**What it measures**: Average absolute difference between predicted and actual, in units.

```
MAE = mean(|actual - predicted|)
```

- Range: 0 to infinity (lower is better)
- Units: same as Quantity_Sold (number of servings/units)

**Interpretation for this project**: A weekly MAE of ~2.08 means each item-week prediction is off by about 2 units on average. For daily, MAE of ~0.39 means each item-day prediction is off by roughly 0.4 units.

**When to use this**: MAE is useful for understanding absolute magnitude of errors. A MAE of 2 units/week per item means your safety stock should cover at least 2 extra units per item per week.

---

### Median Week/Day Accuracy

**What it measures**: The median accuracy across all (item, period) pairs where actual demand was 2 or more units.

```
For each (item, week) where actual >= 2:
    accuracy = 100 * (1 - |actual - predicted| / actual)
Median accuracy = median of all such accuracies
```

- Range: 0% to 100% (higher is better)
- Only counts periods with actual >= 2 units (filters out trivially easy zero/one-sale periods)
- Median (not mean) is used so it is not skewed by a few terrible predictions

**Interpretation for this project**:
- Weekly median of 83%: half of all item-weeks with real demand (2+ units) are predicted within 17% error
- Daily median of 75%: half of all item-days with real demand are predicted within 25% error

**Why median, not mean**: Using the median ensures that a few wildly inaccurate predictions (e.g., predicting 10 when actual is 2) do not inflate the average. The median tells you: "for a randomly chosen item in a randomly chosen week, what accuracy can you expect?"

**Why actual >= 2 filter**: Without this filter, daily accuracy appears artificially high because most item-days have 0 or 1 unit sold, and predicting 0 or 1 gives 100% accuracy. These trivially correct predictions do not reflect the model's real usefulness for supply ordering.

---

### Weeks/Days within +/-20%

**What it measures**: Percentage of (item, period) pairs where actual >= 2 and the prediction is within 20% of actual.

```
For each (item, week) where actual >= 2:
    is_accurate = |actual - predicted| / actual <= 0.20
Pct within ±20% = count(is_accurate) / count(total) * 100
```

- Range: 0% to 100% (higher is better)
- Same actual >= 2 filter as median accuracy
- ±20% means: if you sell 10 units, predicting 8-12 is considered accurate

**Interpretation for this project**:
- Weekly 63%: in roughly 6 out of 10 item-weeks, the model is within ±20% of actual demand
- Daily 49%: roughly half of all item-days are within ±20%

**What this means for operations**: The ~37% of weeks outside ±20% are where you need safety stock or manual adjustment. For weekly supply ordering, this is a reasonable baseline.

---

### Weeks/Days within +/-50%

**What it measures**: Same as above but with a 50% threshold.

- ±50% means: if you sell 10 units, predicting 5-15 is considered "close enough"
- This is a looser threshold that captures whether the model is in the right ballpark

**Interpretation for this project**:
- Weekly 91%, Daily 94%: the model is almost always within ±50% of actual demand
- The remaining ~9% are the worst-case outliers where predictions are completely off

---

## ABC Analysis

Items are classified into three tiers based on cumulative sales volume:

| Class | Definition | Typical Items |
|-------|-----------|---------------|
| **A** | Top items contributing to 70% of total volume | Kopi Susu, Lychee Tea, Black, Air Mineral |
| **B** | Next items contributing to 70-90% of total volume | Nasi Goreng, Tempe Mendoan, Matcha |
| **C** | Remaining items contributing to 10% of total volume | Nirmala, Magic, Menawan |

### ABC BY CLASS

Shows wMAPE and median accuracy broken down per class:

```
A-Class | Items: 27 | wMAPE:  21.5% | Med.W:  83.3%
B-Class | Items: 16 | wMAPE:  20.1% | Med.W:  83.3%
C-Class | Items: 18 | wMAPE:  19.6% | Med.W:  83.3%
```

- **wMAPE per class**: Error rate within that class only. C-class wMAPE can be higher per-unit but matters less for total volume.
- **Median accuracy per class**: How well the model predicts each tier.

**What to watch for**: If A-class wMAPE is significantly worse than B/C, the model struggles with your best sellers (most impactful problem). If C-class is worse, it is less concerning because those items contribute less to total ordering cost.

---

### TOP 10 CLASS A ITEMS

Shows cumulative actual vs predicted quantities over the entire test period for the top 10 best-selling items:

```
  Kopi Susu Husgendam Ice   Actual:    521  Pred:    463  Acc:  88.9%
  Husgendam Platter         Actual:    223  Pred:    171  Acc:  76.7%
```

- **Acc** here is per-item cumulative accuracy: `100 * (1 - |actual_total - pred_total| / actual_total)`
- This is useful for spotting which specific items the model struggles with
- Items below 80% accuracy deserve attention in supply planning

---

## Overfitting Check (scripts/06_check_overfitting.py)

Separate from the evaluation report, the overfitting check compares train vs test metrics:

```
GLOBAL MODEL
  r2                       0.9465     0.8922    -0.0543
  wmape                   13.67      16.41     +2.74

PER-ITEM MODELS (38 items, blend a=0.15)
  r2                       0.9434     0.8668    -0.0766
  wmape                   14.72      18.54     +3.82
```

### What the Gap means

| Gap size | Interpretation |
|----------|---------------|
| **R2 gap < 0.03** | Healthy. Model generalizes well. |
| **R2 gap 0.03 - 0.05** | Acceptable. Minor overfitting, normal for small datasets. |
| **R2 gap 0.05 - 0.15** | Moderate overfitting. Model memorizes some patterns that do not repeat. |
| **R2 gap > 0.15** | Significant overfitting. Model is unreliable for real-world use. |

### Key columns

- **Train**: Metrics computed on training data (model has seen these patterns)
- **Test**: Metrics computed on holdout data (model has NOT seen these)
- **Gap**: Test - Train. Negative R2 gap = test is worse (expected). Negative wMAPE gap = test error is higher (expected).

### Why train R2 should be lower than test R2 in an ideal world

In practice, train R2 is always higher than test R2. The smaller the gap, the more trustworthy the model is for future predictions. Our model uses early stopping, regularization, and blending (global + per-item) to minimize this gap.

---

## Practical Guide: What Numbers Are "Good Enough"?

| Metric | Acceptable | Good | Caution |
|--------|-----------|------|---------|
| **Global wMAPE** | < 30% | < 20% | > 35% |
| **Median Week Accuracy** | > 70% | > 80% | < 60% |
| **Weeks within ±20%** | > 50% | > 65% | < 40% |
| **R2 train-test gap** | < 0.10 | < 0.05 | > 0.15 |

### Bottom line for supply ordering

For a small cafe with 61 menu items:
- **Weekly frequency** is recommended for ordering (less noise, more actionable)
- **wMAPE ~21%** means plan for ±20-25% safety margin on your orders
- **63% of weeks within ±20%** means about 1 in 3 weeks will need manual correction
- Focus on A-class items (top 27) as they drive most of your ordering cost
