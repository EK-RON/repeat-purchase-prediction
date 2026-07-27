# Repeat-Purchase Prediction

Will this customer order again in the next 30 days? An end-to-end propensity model
on 4,334 real customers — built around a **leakage-free temporal design**, judged by
**the metric the business decision actually needs**, and measured against a baseline
that a competent analyst could build in an afternoon.

```bash
pip install -r requirements.txt
python -m src.pipeline      # data, snapshots, CV, holdout, figures — one command
pytest                      # 27 tests, no network, ~4s
```

Fourth in a series on the same dataset:
[`online-retail-eda`](https://github.com/EK-RON/online-retail-eda) cleaned it,
[`retail-dimensional-model`](https://github.com/EK-RON/retail-dimensional-model)
modelled it, [`scraping-pipeline`](https://github.com/EK-RON/scraping-pipeline)
collected a dataset from scratch, and this one predicts from it.

---

## The problem, in one paragraph

> For each customer with at least one prior order, predict whether they will place
> another order within **30 days** of a monthly cutoff date.

The 30-day horizon is a business choice, not a statistical one: the retention campaign
runs monthly, so a prediction reaching beyond the next campaign cycle cannot be acted
on. Everything downstream follows from that sentence — including which metric matters.

---

## Result

**Holdout: October and November 2011, never seen during model selection.**

| Metric | XGBoost | Recency rule | Difference |
|---|---:|---:|---:|
| PR-AUC | **0.619** | 0.440 | +0.180 |
| Precision@20% | **0.646** | 0.485 | +0.161 |
| Lift@20% | **2.07x** | 1.56x | +0.52 |
| Brier | **0.176** | 0.284 | −0.109 |
| Accuracy | 0.749 | 0.683 | — |
| *Accuracy of "nobody ever returns"* | | *0.688* | |

Contacting the top 20% of scored customers reaches **41% of everyone who will return**,
at **2.07x** the hit rate of contacting 20% at random.

---

## Three things this project is actually about

### 1. The temporal design — why the model can't cheat

Most repeat-purchase notebooks build features from the whole dataset, add a label, and
split at random. Both halves of that are wrong. Features computed over the full period
have already seen the outcome, and a random split lets the model learn from October to
predict June — something production can never do.

The structure here makes cheating impossible rather than merely discouraged:

```
        features from here          label from here
    ├──────────────────────────┤ ├─────────────────┤
    ...transaction history...  cutoff   +30 days
```

A **snapshot** is one `(customer, cutoff)` pair. Nine monthly cutoffs from March to
November produce **23,300 rows and 33 features**. Features come from
`transactions[date < cutoff]`; labels from `transactions[cutoff <= date < cutoff + 30d]`.
Two separate functions build them, and neither ever receives the other's rows.

That guarantee is usually a comment in a notebook. Here it is a test:

```python
def test_features_are_identical_when_the_future_is_deleted(transactions):
    with_future = snapshot(transactions, CUTOFF)
    without_future = snapshot(transactions[transactions.date < CUTOFF], CUTOFF)

    pd.testing.assert_frame_equal(with_future[features], without_future[features])
    assert without_future["label"].sum() == 0
```

Delete the entire outcome window; not one feature value moves, and every label goes to
zero. A second test inflates every post-cutoff value 1000x and asserts the same thing.

**Cross-validation expands forward in time.** Each fold trains on cutoffs 1..n and
validates on n+1 — the shape of the real deployment. `KFold` would have been optimistic
and meaningless.

![temporal design](reports/figures/01_temporal_design.png)

The chart above is also the reason a single averaged score misleads: **24% of customers
return in August, 35% in November.** The target itself moves with the season.

### 2. Why accuracy is the wrong metric — demonstrated, not asserted

The base rate is 29%. So a model that predicts *nobody ever returns* scores **68.8%
accuracy** on the holdout.

The recency rule scores **68.3%** — *worse than that*. And yet it is genuinely useful:
it ranks customers well enough to beat random targeting by **1.56x**. Accuracy cannot
see this, because it grades a threshold nobody uses on a decision nobody makes.

The decision being made is: *the campaign has a budget; which customers do we contact?*
That is a **ranking** problem, so the metrics follow:

- **PR-AUC** — the honest summary for an imbalanced ranking task, computed against the
  positive class instead of being propped up by easy negatives.
- **Precision@k and lift@k** — the actual objective. Of the 20% we contact, what share
  return, and how much better is that than random?
- **Brier score and calibration** — because turning a score into a budget decision needs
  a probability that means what it says, not merely a correct ordering.

![PR and ROC](reports/figures/03_pr_roc_curves.png)

One further trap the per-cutoff table exposes: **PR-AUC is not comparable across the two
holdout months** (0.568 in October, 0.665 in November) because it scales with the base
rate. Lift is stable (2.12 and 2.03). Quoting the PR-AUC jump as improvement would be
reading seasonality as skill.

![gains](reports/figures/04_gains.png)

### 3. The honest model comparison

| Model | CV PR-AUC | ± std | Holdout PR-AUC |
|---|---:|---:|---:|
| Random forest | 0.607 | 0.011 | 0.616 |
| XGBoost | 0.603 | 0.011 | **0.619** |
| Logistic regression | 0.597 | 0.009 | 0.612 |
| Recency rule | 0.423 | 0.026 | 0.440 |

**The three models are separated by less than one fold-to-fold standard deviation.**
Cross-validation ranked random forest first; the holdout ranked XGBoost first. That
reordering *is* the finding: the gap between them is noise, and reporting "XGBoost wins"
without the spread would be reading a coin flip as a result.

What is not noise is the gap to the baseline: **PR-AUC 0.42 → 0.62**, lift 1.56 → 2.07.
The models earn their complexity; they just don't earn it from each other.

Logistic regression landing within 0.02 of gradient boosting is itself informative. It
says the signal in these features is largely monotonic — which is a credit to the
feature engineering rather than a failure of the models. XGBoost hyperparameters were
tuned by grid search *inside the temporal CV* (`tune_xgboost`), never against the
holdout; the search moved PR-AUC from 0.596 to 0.603.

![cv results](reports/figures/02_cv_results.png)

---

## What the model learned

![permutation importance](reports/figures/06_permutation_importance.png)

Permutation importance on the **holdout**, scored by PR-AUC — not the tree's built-in
`gain`, which is biased towards high-cardinality features and says nothing about unseen
data.

The top feature is `recency_vs_gap`, which is engineered rather than raw:

```python
recency_vs_gap = days_since_last_order / mean_days_between_orders
```

Forty days of silence means nothing on its own. Forty days of silence from someone who
normally returns every twenty is a customer slipping away. Raw `recency_days` ranks
seventh; the ratio that puts it in context ranks first.

At the other end, `is_uk`, `avg_items` and `avg_order_value` score *negative* importance
— shuffling them slightly improves the score, which is the signature of noise. They are
reported rather than quietly dropped.

---

## Calibration, and where it breaks

![calibration](reports/figures/05_calibration.png)

Well calibrated in the top deciles, under-confident at the bottom. The stability check
explains why:

| Cutoff | Base rate | Mean predicted | PR-AUC | Lift@20% |
|---|---:|---:|---:|---:|
| 2011-10-01 | 0.274 | 0.282 | 0.568 | 2.12 |
| 2011-11-01 | 0.347 | 0.292 | 0.665 | 2.03 |

October is nearly perfect: predicted 28.2%, observed 27.4%. **November is not: the model
predicts 29.2% against an observed 34.7%.** It was trained on March–September, whose base
rates average 29%, and November is the Christmas run-up.

The ranking survives — lift barely moves — but the probabilities do not. In deployment
that matters: an expected-value calculation using these probabilities would under-budget
the Christmas campaign by roughly a sixth. The fix is periodic recalibration against
recent outcomes, not a better model.

---

## From a score to a decision

![capacity](reports/figures/07_capacity_tradeoff.png)

| Contact | Precision | Recall | Net value | Per contact |
|---:|---:|---:|---:|---:|
| 5% | 0.887 | 0.142 | £8,077 | £23.34 |
| 10% | 0.802 | 0.257 | £14,529 | £20.97 |
| 20% | 0.646 | 0.414 | £22,983 | £16.59 |
| 50% | 0.455 | 0.730 | £38,932 | £11.24 |

There is a real tension here that the model cannot resolve: **total value keeps rising
as you contact more people, while value per contact keeps falling.** Whether that means
5% or 50% depends on whether the constraint is money or attention — a business decision
the model exists to inform, not to make.

These figures rest on assumptions stated in `config.py` (£1.50 per contact, £28
incremental margin). The one doing the most work is that contacting someone who would
have returned anyway still earns the margin. That is generous: a real programme needs a
**holdout group to measure incrementality**, otherwise it pays to reach people who were
already coming back. The number above is an upper bound, and it is labelled as one.

---

## Repository layout

```
src/
  config.py     the problem definition, temporal design, business assumptions
  data.py       checksum-pinned source -> clean transaction universe
  features.py   snapshots: 33 features from the past, labels from the future
  metrics.py    precision@k, lift@k, gains, calibration, expected value
  models.py     baseline + 3 classifiers, expanding-window CV, tuning
  evaluate.py   holdout, calibration, permutation importance, stability
  visualize.py  seven report figures
  pipeline.py   CLI that reproduces every number
tests/
  test_features.py  the leakage guarantee, on hand-checkable fixtures
  test_models.py    metric definitions and the validation scheme
reports/          generated results.md, CV folds, holdout metrics, figures
```

---

## Limitations

- **One year of data, one holdout of two months.** Every seasonal conclusion rests on a
  single Christmas. The calibration drift found in November is a real effect in this
  data; whether it generalises needs more years than this dataset has.
- **Propensity is not incrementality.** The model predicts who will return, not who will
  return *because they were contacted*. Those are different targets, and only a
  randomised holdout can measure the second. The expected-value table is explicitly an
  upper bound.
- **Guests are excluded.** A quarter of the source rows have no customer id, so they
  cannot be tracked across snapshots. The model therefore describes the identified
  customer base, not all revenue.
- **The same customer appears in several snapshots.** This mirrors production, where you
  score the same people every month, but it means fold-to-fold scores are not fully
  independent. A customer-disjoint split would answer a different, more academic
  question.
- **No uncertainty on the headline numbers.** Bootstrap confidence intervals on
  precision@k would say more than the point estimate does, particularly given how close
  the three models are.

## Data

"Online Retail", D. Chen (2015), UCI Machine Learning Repository,
[doi.org/10.24432/C5BW33](https://doi.org/10.24432/C5BW33), CC BY 4.0. Pinned by SHA-256
in `src/config.py` — the same checksum as the other projects in this series.
