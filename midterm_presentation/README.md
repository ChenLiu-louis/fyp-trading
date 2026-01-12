## 10-minute Interim Presentation (PPT)

This folder contains a ready-to-copy slide deck script based on your current interim report:
- `presentation_slides.md` (slide content + English speaker notes)
- `presentation_beamer.tex` (compile on Overleaf to get a PDF slide deck)

### How to turn it into a PowerPoint / Google Slides quickly

1. Create a new deck (PowerPoint / Google Slides)
2. Make **10 slides**
3. For each slide in `presentation_slides.md`:
   - Copy the **Slide Content**
   - Insert the suggested image from `outputs/plots/`
4. Rehearse once (aim for ~60s per slide)

### Option B: Compile a PDF "PPT" on Overleaf (Beamer)

1. Create a new Overleaf project
2. Upload:
   - `midterm_presentation/presentation_beamer.tex`
   - the images listed below (copy from `outputs/plots/`)
3. Overleaf settings:
   - Compiler: **pdfLaTeX**
4. Click **Recompile**

### Images used (copy these into your slide project)

- `outputs/plots/classic_rsi_backtest_20251229_220548.png`
- `outputs/plots/classic_dualma10_50_backtest_20251229_220548.png`
- `outputs/plots/lstm2_backtest_20251229_183712.png`
- `outputs/plots/transformer_backtest_20251229_125307.png`
- `outputs/plots/informer_backtest_20251229_144632.png`
- `outputs/plots/informer_opt_backtest_20251229_152746.png`
- `outputs/plots/finbert_portfolio_backtest_20251230_143439.png`


