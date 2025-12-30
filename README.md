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

### Other runnable scripts

```bash
python run_transformer_pipeline.py
python run_informer_pipeline.py
python run_informer_opt_pipeline.py
python run_trading_param_sweep.py
```

### News data (for pretrained text models / FinBERT stage)

Fetch and write news into `data/news/news.csv`:

```bash
python run_fetch_news_gdelt.py
```

Then run the FinBERT multi-ticker pipeline (requires additional deps: `transformers`, `datasets`, `accelerate`, `sentencepiece`):

```bash
python run_finbert_multiticker_pipeline.py
```

News outputs:
- `data/news/news.csv`: required by the FinBERT pipeline, schema: `date,ticker,text`
- `data/news/news_raw_gdelt_*.csv`: raw fetched articles (url/domain/snippet/etc.)

### Notebooks

The original notebooks are kept as references. New strategy notebooks are added for classic quant baselines (MACD/RSI/Bollinger etc.).


