## COMP4913 Interim Presentation (10 minutes)

**How to use this file**
- Each slide is separated by `---`.
- For each slide, copy the **Slide Content** into PowerPoint.
- Insert the suggested image from `outputs/plots/` (same filename).
- Read the **Speaker Notes** as your English script.

---

### Slide 1 — Title

**Slide Content**
- **From Classic Strategies to Time-Series Transformers**
- **A Quantitative Trading System (Interim Milestone)**
- Instrument: **2800.HK** (Hang Seng Index proxy ETF)
- Student: **Liu Chen** | Supervisor: **Prof. Henry C. B. Chan**

**Speaker Notes (≈40s)**
Good morning. Today I will present my interim milestone for a quantitative trading system. The project goal is to build a reproducible, time‑respecting pipeline and to evaluate progressively stronger baselines. I start from classic strategies, then move to LSTM and Transformers, and finally integrate a news module using FinBERT. My main instrument is 2800.HK, which is a liquid ETF tracking the Hang Seng Index and serves as a representative Hong Kong market proxy.

---

### Slide 2 — Agenda (10 minutes)

**Slide Content**
- **1) Motivation & goal** (system milestone)
- **2) Data & evaluation protocol** (no leakage)
- **3) Baselines**: classic → LSTM → Transformer → Informer → Informer‑OPT
- **4) News stage**: GDELT + FinBERT (current limitation)
- **5) Key lessons & next steps**

**Speaker Notes (≈30s)**
I will first explain the motivation and the midterm objective. Then I describe the data and the evaluation protocol that avoids look‑ahead leakage. Next, I summarize results from classic strategies and ML/DL models under the same backtesting rules. After that, I introduce the news stage and why its current performance is not conclusive. Finally, I highlight the key lessons and the next steps.

---

### Slide 3 — Motivation & Midterm Objective

**Slide Content**
- Markets are noisy; persistent alpha is difficult (EMH)
- Midterm focus: **build the research engine**
  - time‑respecting evaluation (**walk‑forward CV**)
  - reproducible outputs (**configs, metrics, plots**)
  - consistent trading rules (**probabilities → positions**)
- Strategy goal: compare models fairly under the same framework

**Speaker Notes (≈60s)**
Financial markets are noisy and are influenced by many factors. Under the Efficient Market Hypothesis, consistent alpha is hard, so the focus must be on rigorous evaluation. My midterm goal is not to claim a final profitable strategy yet. Instead, it is to build a research engine: a pipeline that is time‑respecting, reproducible, and produces standardized artifacts. This allows me to iterate quickly and compare models fairly. The key is to keep the data processing, cross‑validation, and backtest rules consistent so that differences in results mainly reflect differences in the models, not differences in evaluation.

---

### Slide 4 — Data, Labels, and Evaluation (No Look‑Ahead)

**Slide Content**
- Data: **daily OHLCV** from `yfinance` (2800.HK)
- Target: next‑day **log return** \( r_{t+1}=\log(C_{t+1}/C_t) \)
- Labels: **volatility‑adaptive 3‑class**
  - Up / Neutral / Down using dynamic threshold
- Evaluation: **fixed‑window walk‑forward CV**
  - train → val → test, rolling forward
  - scaler fit on train only

**Speaker Notes (≈70s)**
I use daily OHLCV data from yfinance. The prediction target is the next‑day close‑to‑close log return. For classification, I use a three‑class label: Up, Neutral, and Down. Importantly, the threshold is volatility‑adaptive: it scales with recent rolling volatility, which reduces regime sensitivity and avoids using a fixed arbitrary threshold. For evaluation, I use fixed‑window walk‑forward cross‑validation. Each fold trains on a past window, validates on the next window, and tests on a future window. Feature standardization is fit only on the training split and then applied to validation and test splits. This design helps prevent data leakage.

---

### Slide 5 — From Probabilities to Trades (Backtest Layer)

**Slide Content**
- Model outputs \( P(\text{Down}), P(\text{Neutral}), P(\text{Up}) \)
- **Confidence threshold**: if max prob < thr → Neutral (no trade)
- Map to position: **Long / Short / Flat**
- Constraints:
  - **min holding period** (reduce flip‑flop)
  - **transaction cost**: 2 bps on position changes
- Why it matters: trading rule can dominate realized P&L

**Speaker Notes (≈70s)**
All ML/DL models output class probabilities for Down, Neutral, and Up. I convert those probabilities into trades using a consistent translation layer. First, I apply a confidence threshold: if the model is not confident enough, I force the prediction to Neutral and do not take a directional position. If it is confident, I map the class to a discrete position: long, short, or flat. I also enforce a minimum holding period to reduce frequent position flipping, and I include transaction costs of 2 basis points when the position changes. A key lesson from this project is that this translation layer—thresholds, holding rules, and costs—can dominate realized performance, even if the underlying classifier accuracy changes only slightly.

---

### Slide 6 — Classic Strategy Baselines (Interpretability First)

**Slide Content**
- Purpose:
  - validate backtest mechanics
  - build intuition for 2800.HK
  - create transparent benchmarks
- Representative baselines:
  - **RSI mean‑reversion**
  - **Dual Moving Average trend**
- Key takeaway: some strategies have high Sharpe but **low coverage**

**Suggested Image**
- `outputs/plots/classic_rsi_backtest_20251229_220548.png` (and/or)
- `outputs/plots/classic_dualma10_50_backtest_20251229_220548.png`

**Speaker Notes (≈60s)**
Before any ML training, I tested classic strategies to validate the backtesting pipeline and build intuition for 2800.HK. I focus on interpretable rules like RSI mean‑reversion and dual moving average trend following. One important observation is the difference between risk‑adjusted metrics and total return. For example, RSI can show a high Sharpe ratio but trades very rarely, leading to low market exposure and lower total return compared with buy‑and‑hold. These baselines provide transparent reference points for later ML/DL experiments.

---

### Slide 7 — LSTM Baseline (First End‑to‑End ML Run)

**Slide Content**
- Input: sequences of technical features (lookback=30, 16 features)
- Model: **LSTM classifier** (PyTorch)
- Backtest (last 252 OOS days):
  - Total return: **−6.23%**
  - Coverage: **25.4%**
  - Sharpe: **−0.57**

**Suggested Image**
- `outputs/plots/lstm2_backtest_20251229_183712.png`

**Speaker Notes (≈70s)**
The first ML milestone is an LSTM classifier trained on sequences of technical indicators. Each sample is a 30‑day window of features, and the model predicts the 3‑class direction label. This run is important because it produced a full set of standardized artifacts—cross‑validation metrics, out‑of‑sample probabilities, and backtest results—so the pipeline is reproducible. In the last 252 out‑of‑sample trading days, the strategy underperformed buy‑and‑hold, with about minus 6 percent total return and only about 25 percent coverage. Although the performance is not strong, the key achievement is that the full ML research workflow is working end‑to‑end.

---

### Slide 8 — Transformer Encoder Baseline (GPU Run)

**Slide Content**
- Same protocol as LSTM (fair comparison)
- Model: **Transformer encoder** (self‑attention)
- Backtest (last 252 OOS days):
  - Total return: **+0.94%**
  - Coverage: **62.3%**
  - Sharpe: **0.15**

**Suggested Image**
- `outputs/plots/transformer_backtest_20251229_125307.png`

**Speaker Notes (≈70s)**
Next, I implemented a Transformer encoder baseline and evaluated it under the same protocol to ensure a fair comparison. The Transformer increases model capacity and uses self‑attention to capture patterns across the sequence without recurrence. In the last 252 out‑of‑sample days, the strategy achieved a small positive return around 1 percent, with higher coverage around 62 percent. However, it still trails buy‑and‑hold in total return and has modest risk‑adjusted performance. This suggests that the model alone is not sufficient; the interaction between probability calibration, trade mapping, and costs becomes critical.

---

### Slide 9 — Informer & Informer‑OPT (Key Milestone: Trading Layer Matters)

**Slide Content**
- Informer‑style: ProbSparse attention + distilling (classification adaptation)
- Initial Informer‑style (252 OOS days):
  - Total return: **−1.02%**
  - Coverage: **42.1%**
- Informer‑OPT (feature expansion + longer context + rule sweep):
  - Total return: **36.75%** vs Buy&Hold **36.92%**
  - Sharpe: **1.95**, Max DD: **−7.51%**
  - Trading rule: **thr=0.36, hold≥5, long‑only**

**Suggested Image**
- `outputs/plots/informer_backtest_20251229_144632.png` (and)
- `outputs/plots/informer_opt_backtest_20251229_152746.png`

**Speaker Notes (≈80s)**
I then explored an Informer‑style model, which is designed for long sequence efficiency using ProbSparse attention and distilling. Since my task is classification rather than multi‑step forecasting, I adapt it as an encoder classifier. The initial Informer‑style run still underperformed. The important turning point is Informer‑OPT: I expanded the feature set, increased temporal context, used full 3‑class loss for better probability calibration, and then swept the trading‑rule parameters without retraining the model. This produced a result that almost matches buy‑and‑hold over the same window, with lower drawdown and fewer trades. This strongly supports the lesson that the probability‑to‑trade translation layer—threshold and holding constraints—can dominate realized trading performance.

---

### Slide 10 — News Stage (GDELT + FinBERT) + Next Steps

**Slide Content**
- Goal: add **external signal** beyond price/indicators
  - 2800.HK is index ETF → macro/news can influence flows
- Pipeline:
  - Fetch news via **GDELT DOC API**
  - Aggregate to daily text per (date, ticker)
  - Fine‑tune **FinBERT** for 3‑class direction label
  - Portfolio backtest (equal‑weight HK universe)
- Current limitation: **short OOS overlap** (coverage issue)
- Next steps:
  - expand news coverage + stable fine‑tuning
  - add **PatchTST** baseline under same protocol

**Suggested Image**
- `outputs/plots/finbert_portfolio_backtest_20251230_143439.png`

**Speaker Notes (≈80s)**
Finally, I integrated a news stage to add an external signal beyond technical indicators. Since 2800.HK tracks the broader Hong Kong equity market, macro and market news can influence sentiment and flows. The pipeline fetches news via the GDELT Document API, aggregates multiple articles into a daily text per ticker, and then fine‑tunes FinBERT to predict the next‑day 3‑class direction label. I evaluate the signal using an equal‑weight HK portfolio backtest with transaction costs. At the moment, the main limitation is data coverage: the overlap between news dates and price labels in the test split is too short, so performance is not statistically meaningful yet. My next steps are to expand and stabilize news coverage, and to add a stronger recent time‑series baseline like PatchTST under the same evaluation protocol.


