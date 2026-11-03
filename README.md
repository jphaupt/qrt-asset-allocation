# QRT Asset Allocation Performance forecasting Challenge

"Challenge Data" task, by QRT. challengedata.ens.fr/challenges/167

## Description

Competitive challenge

Event started on Jan. 22, 2026. (I started Oct 19, 2026)

Description below taken from website.

### Challenge context

#### Trust or Short? Predicting the Performance of daily Asset Allocations

In the world of systematic trading, asset allocations are everywhere — but signal quality is everything.

Each day, traders are flooded with candidate allocations: portfolio constructions based on recent signals, liquidity flows, or historical patterns. Some of these portfolios will perform well in the next trading session. Others will underperform—or worse, underperform so consistently that shorting them might be the more profitable move.

This challenge centers around a simple yet high-stakes question:
**Can you predict whether a given asset allocation is worth following — or shorting?**

#### What is an Asset Allocation?

An asset allocation is a systematic method of constructing a portfolio of assets using predefined signals or rules. In this challenge, each allocation is defined by a set of portfolio weights, that can be positive or negative, applied on a specific day and held for one trading session. From one day to another, an allocation can rebalance its weights to a certain proportion, called turnover, depending on the rules used to compute it. The returns of an asset allocation reflect the aggregated performance of these weighted positions rebalanced every day.

#### Mathematical Definition

For a given trading day $t$, an allocation $S$, and $M$ assets in a trading universe, let the weights of allocation $S$ at time $t$ be:

```math
w_{S,t} = (w_{S,t,1}, w_{S,t,2}, \dots, w_{S,t,M})
```

The performance, also often referred to as return, of asset $i$ from day $t$ to day $t+1$:

```math
r_{i,t+1}
```

Then the **realized return** of an allocation $S$ at $t+1$ is given by:

```math
r_{S,t+1} = \sum_{i=1}^{M} w_{S,t,i} \, r_{i,t+1}
```

### Challenge goals

Each row in the dataset represents **a day *and* an asset allocation**, materialized as a portfolio constructed and rebalanced on that day. We give you a **history of how that allocation behaved when rebalanced** over the past 20 trading days: its daily performance, liquidity behaviour (proxied through weighted volumes), and median turnover.

The goal is to use that historical footprint to predict the **sign of the allocation's performance on the following day**.

- If the model predicts positive return → trust the allocation.
- If the model predicts negative return → short the allocation.

#### Evaluation Metric

You will be evaluated based on **accuracy**, which measures how often the model correctly predicts the **direction** (sign) of an allocation's next-day return.

For each row $i$ indexed by a time stamp $t$ and an allocation $S$, we provide the **true returns** $r_i$ for the following trading day. Your model must predict the **sign** of that return:

- `1` if you believe the allocation will have a **positive return** (go long)
- `0` if you believe the return will be **negative**.

Only the **sign** is evaluated — not your capacity to predict the return's magnitude.

```math
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[ \operatorname{sign}(\hat{r}_i) = \operatorname{sign}(r_i) \right]
```

```math
\text{Accuracy} = \frac{1}{T \cdot M} \sum_{t=1}^{T} \sum_{S=1}^{M} \mathbb{1}\left[ \operatorname{sign}(\hat{r}_{S,t+1}) = \operatorname{sign}(r_{S,t+1}) \right]
```

Where:

- $N$ is the **number of rows**
- $\operatorname{sign}(x) = 1$ if $x > 0$, else $0$
- $\mathbb{1}$ is the **indicator function** (equals 1 if the condition is true, 0 otherwise)
- The **true next day return** for allocation $S$ at timestamp $t$: $r_i = r_{S,t+1}$
- The **predicted next day return** for allocation $S$ at timestamp $t$: $\hat{r}_i = \hat{r}_{S,t+1}$

### Data description

The dataset is formatted as a time series with a multi-index of (date, allocation). Each row contains:

- **20-day history of allocation returns**
- **20-day history of volume-weighted liquidity behavior**
- **Allocation median turnover**
- **Allocation anonymized GROUP**
- **Next day allocation return**: the performance of the allocation on the next day. **You are given the true performance for training, but you will be only evaluated on your capacity to predict its sign.**

#### Columns

- `TS` — Timestamp of the snapshot (dates were anonymized and shuffled, so there is no guarantee of continuity even if the labels are called `DATE_0001`, `DATE_0002`, `DATE_0003`)
- `ALLOCATION` — Name of the allocation (`ALLOCATION_01` is the same for `DATE_0001`, `DATE_0002`, etc.)
- `RET_{i}` for i in 1, ..., 20 — Allocation's return on last day i
- `SIGNED_VOLUME_{i}` for i in 1, ..., 20 — Allocation's signed volume on last day i (see definition below)
- `MEDIAN_DAILY_TURNOVER` — Allocation's median daily turnover (see definition below)
- `GROUP` — Anonymized allocation group
- `TARGET` — Allocation's true next day return

#### Volumes, allocation weights, and turnover

At every day $t$, each allocation $S$ follows this property, given a universe of $M$ trading instruments:

```math
\forall\, t,\ \forall\, S:\quad \sum_{i=1}^{M} \lvert w_{S,i,t} \rvert = 1
```

The **`SIGNED_VOLUME`** of an allocation $S$ at $t$ is given by:

```math
V_{S,t} = \sum_{i=1}^{M} w_{S,t,i} \, V_{i,t}
```

where $V_{i,t}$ is the total volume traded on the market of asset $i$ during the trading session at timestamp $t$. For homogeneity, these $V_{S,t}$ were rescaled in a rolling fashion to ensure comparability across different styles of allocations.

The **`MEDIAN_DAILY_TURNOVER`** of an allocation $S$ at $t$ is given by:

```math
TO_{S,t} = \text{TURNOVER}_{S,t} = \sum_{i=1}^{M} \lvert w_{S,t,i} - w_{S,t-1,i} \rvert
```

```math
MDT_{S,t} = \operatorname{median}(TO_{S,t}, TO_{S,t-1}, \dots, TO_{S,t-20})
```

#### Files

**All files are indexed by a unique `ROW_ID`, referring to a unique tuple (date, allocation), allowing you to map `X_train` with `y_train`.**

- **`X_train.csv`** — the training set features
- **`y_train.csv`** — the training set target
- **`X_test.csv`** — the test set features
- **`sample_submission.csv`** — a random submission file in the correct format
- **`benchmark_submission.ipynb`** — a benchmark submission notebook to generate the benchmark you see in the leaderboard

527,073 observations (i.e. lines) are available in the train set, while 31,870 observations are in the test set.

### Benchmark description

**The data section contains a Benchmark Notebook generating the benchmark submission you see in the leaderboard. It also shows how to correctly generate a submission file.**

The notebook contains:

- additional features representing the average historical performance of the allocations on multiple windows
- additional features representing the average historical performance of all allocations on multiple windows
- additional features representing the historical volatility of each allocation on the past 20 days
- additional features representing the average historical volatility of all allocations on the past 20 days
- a ridge model fitted on all + additional features
- a LightGBM model fitted on all + additional features, with a cross-validation section (benchmark submitted)

**The public score, visible on the leaderboard, of the LightGBM model is 0.5079.**

### Challenge provider

Qube Research & Technologies (QRT) is a global quantitative and systematic investment manager, operating in all asset classes across the world. Driven by a shared passion for data, research, technology and trading expertise, they strive to deliver high-quality returns for their investors. Established in 2018, QRT can rely on its 2,000+ employees across 16 offices globally. QRT supports various coding initiatives as well as academic projects developing and promoting maths and science education.
