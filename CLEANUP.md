## Cleanup / Refactor notes (Dec 2025)

This project started as several exploratory notebooks. To support your mid-term report and the later system-integration phase, the key logic is now modularized under `fyp_trading/` and runnable via:

- `python run_lstm2_pipeline.py`

### What to keep (recommended)

- `run_lstm2_pipeline.py`: reproducible ML pipeline; writes artifacts to `outputs/`
- `fyp_trading/`: reusable modules for data/features/labeling/model/cv/backtest/report
- Classic strategy notebooks:
  - `strategy_macd.ipynb`
  - `strategy_rsi.ipynb`
  - `strategy_bollinger.ipynb`
  - `strategy_compare.ipynb`

### Legacy notebooks (kept for reference)

These notebooks contain overlapping versions of similar ideas. They are **kept** to avoid losing work, but are now effectively superseded by the modular code.

- `LSTM_2.ipynb`: original end-to-end ML pipeline + vectorized backtest prototype
- `LSTM_fixed.ipynb`: fixed-window LSTM baseline (cls+reg); good as earlier reference
- `LSTM.ipynb`, `LSTM_3.ipynb`: later experimental/improved versions (labeling/risk ideas)
- `xgboost_fixed.ipynb`, `xgboost_CVGrid.ipynb`: XGBoost baselines
- `demo1.ipynb`: early exploratory calculations/plots

### Safe deletion policy

If you want to **actually delete** legacy notebooks, do it only after confirming:
- the modular pipeline reproduces the key plots/metrics you need for your report
- any unique idea in a notebook has been migrated into modules or documented


