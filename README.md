## FYP Quant Trading (modularized)

You currently have several exploratory notebooks (XGBoost/LSTM). This repo now adds a **reusable Python module** under `fyp_trading/` so you can:
- run experiments reproducibly
- automatically save model + plots + reports under `outputs/`
- compare ML vs classic strategies in your mid-term report

### Quick start: reproduce the LSTM_2 pipeline outputs

Run:

```bash
python run_lstm2_pipeline.py
```

Artifacts:
- `outputs/models/`: saved last-fold model + scaler stats
- `outputs/plots/`: backtest equity/position/probability plot
- `outputs/reports/`: CV metrics, CV predictions, backtest time series, backtest stats JSON, run config snapshot

### Notebooks

The original notebooks are kept as references. New strategy notebooks are added for classic quant baselines (MACD/RSI/Bollinger etc.).


