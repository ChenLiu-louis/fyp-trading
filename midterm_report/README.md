## Midterm Report (Overleaf / LaTeX)

This folder contains `中期报告.tex`, written in **English** and designed to be compiled with **XeLaTeX** (better fonts and future multilingual support).

It also contains an AAAI-style version:
- `midterm_report_aaai.tex` (AAAI 2026 template style; **PDFLaTeX only**)

### How to compile on Overleaf

1. Create a new Overleaf project
2. Upload either:
   - `midterm_report/中期报告.tex` (XeLaTeX report style), OR
   - `midterm_report/midterm_report_aaai.tex` (AAAI style)
3. Upload the backtest plot produced by your run:
   - `outputs/plots/lstm2_backtest_20251229_183712.png`
4. Select the compiler:
   - For `中期报告.tex`: **Menu → Settings → Compiler → XeLaTeX**
   - For `midterm_report_aaai.tex`: **Menu → Settings → Compiler → pdfLaTeX**
5. Click **Recompile**

If you rename the figure, update the `\\includegraphics{...}` filename accordingly.


