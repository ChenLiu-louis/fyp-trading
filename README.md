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

News outputs:
- `data/news/news.csv`: required by sentiment pipelines, schema: `date,ticker,text`
- `data/news/news_raw_gdelt_*.csv`: raw fetched articles (url/domain/snippet/etc.)

### Informer + FinBERT Sentiment Pipeline (Recommended)

Uses **pre-trained FinBERT as a fixed feature extractor** (no fine-tuning required).
FinBERT extracts sentiment probabilities (P_neg, P_neu, P_pos) from news headlines,
which are then concatenated with technical features and fed into Informer.

```bash
# Default: use ALL market news for 2800.HK (since it's an ETF)
python run_informer_sentiment_pipeline.py

# Run baseline without sentiment for comparison
python run_informer_sentiment_pipeline.py --no-sentiment

# Only use ticker-specific news
python run_informer_sentiment_pipeline.py --ticker-news-only

# Change fill method for days without news (default: zero)
python run_informer_sentiment_pipeline.py --fill-method ffill
```

Outputs saved to `outputs/reports/`, `outputs/plots/`, `outputs/models/`:
- `informer_sent_backtest_stats_*.json`: backtest results with sentiment
- `informer_sent_backtest_*.png`: equity curve visualization
- `data/news/sentiment_cache_*.csv`: cached FinBERT sentiment (reused on subsequent runs)

### FinBERT Fine-tuning Pipeline (Alternative)

Fine-tunes FinBERT on price-labeled news data (requires more news coverage):

```bash
python run_finbert_multiticker_pipeline.py
```

Requires additional deps: `transformers`, `datasets`, `accelerate`, `sentencepiece`

### Notebooks

The original notebooks are kept as references. New strategy notebooks are added for classic quant baselines (MACD/RSI/Bollinger etc.).


