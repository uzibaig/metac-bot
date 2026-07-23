# Superforecaster Bot — Design & Roadmap

Goal: build a superforecasting AI bot (à la FutureSearch / Preseen), compete on
benchmarks first (Metaculus AI Benchmark tournament), productize later only once
the track record is proven.

## Why this approach works

These systems are scaffolding around frontier LLMs, not custom models. The
research (Halawi et al. 2024; ForecastBench) shows most of the gain over a raw
LLM comes from four things:

1. **Retrieval** — good, current research fed into the prompt.
2. **Base-rate-first reasoning** — outside view before inside view (Tetlock's #1 rule).
3. **Ensembling** — aggregating diverse, independent estimates (trimmed mean of
   log-odds, slight extremization).
4. **Calibration** — post-hoc correction learned from the bot's own resolved
   forecasts.

State of the art (mid-2026): superforecasters ~0.081 difficulty-adjusted Brier on
ForecastBench vs ~0.101 best raw LLM; scaffolded bots (Preseen, FutureSearch) are
at rough parity with top humans in Metaculus tournaments.

## Target pipeline (replaces the template's single-shot forecast)

```
Question → 1. Triage & decomposition (parse resolution criteria, Fermi-decompose)
         → 2. Outside view: reference-class base rate BEFORE reading news
         → 3. Inside view: N parallel research agents with diverse personas
              (news-focused / primary-source / contrarian / market-price checker)
         → 4. Synthesis: Bayesian update of prior + log-odds aggregation,
              price in resolution risk
         → 5. Calibration layer (learned from own Brier history; clamp extremes)
         → 6. Output: probability + full reasoning trail + sources
         → 7. Scoring DB: store every forecast, score on resolution,
              feed errors back into 5
```

## Roadmap

- [x] Phase 1a: clone Metaculus/metac-bot-template, install deps
- [x] Phase 1b: keys in .env, local smoke test passed 2026-07-07 — 9/9 forecasts
      submitted to bot-testing-area, $0.15/question, bot account: usb.18
- [x] Phase 1c: fork live at github.com/uzibaig/metac-bot, secrets added,
      Actions enabled; Test Bot run succeeded 2026-07-08 (verified on Metaculus).
      Bot now runs every 20 min on the live tournament automatically.
- [x] Re-forecast policy (2026-07-22, live template bot, main.py): the
      tournament run now re-forecasts a previously-forecasted question only
      if it's long-dated (>= 30 days to close) AND enough time has passed
      since the last forecast (>= 7 days) — instead of forecasting once and
      never touching it again. Implements "perpetual beta" belief updating;
      Metaculus scores forecasters on accuracy across a question's whole
      open lifetime, so a stale forecast on a question resolving months out
      costs real score. Tunable via REFORECAST_MIN_DAYS_TO_CLOSE /
      REFORECAST_MIN_DAYS_SINCE_LAST_FORECAST at the top of main.py's
      dispatch block. Note: had to read forecast timestamps from the raw
      api_json instead of question.previous_forecasts — the library field is
      unreliable because it depends on community-prediction visibility,
      which Metaculus hides for this account on most open questions
      (unrelated bug affecting an unrelated field). Unit-tested (7/7 pass)
      since no live questions were open to validate against at ship time.
- [~] Phase 2 (in progress): `superforecaster_bot.py` v2 built —
      6 research lenses (news, base rates, skeptic/fine-print, market signals,
      causal mechanism, trend extrapolation), 7 personas (Bayesian anchorer,
      status-quo, red team, Fermi, resolution lawyer, pre-mortem, scenario
      partitioner) cycled across 3 model families (gpt-4o, claude-sonnet-5,
      gemini-2.5-pro via OpenRouter), extremized log-odds pooling.
      Publishing stays OFF for this bot until it beats the template.
      **Evaluation**: community-prediction benchmarking (`benchmark.py`) is
      blocked — Metaculus hides the CP on open questions from accounts that
      haven't predicted on them (only 1 CP-visible open binary question
      account-wide). Instead: shadow A/B against real resolutions —
      `superforecaster_bot.py --mode minibench --limit N` (unpublished, saved
      to reports/) vs the live template's published forecasts; score with
      `compare_forecasts.py` once questions resolve (MiniBench = days).
- [ ] Phase 3: scoring DB (SQLite), calibration layer, continuous re-forecasting
      of open questions
- [ ] Phase 4 (only after accuracy proven): user-facing app / API with public
      track record page

## Key template facts (main.py)

- `SummerTemplateBot2026(ForecastBot)` — override `run_research` and the four
  `_run_forecast_on_*` methods; framework handles Metaculus API, aggregation
  (`predictions_per_research_report=5` samples already ensembled), scheduling.
- Modes: `--mode test_questions` (bot-testing-area), `tournament` (live AIB +
  MiniBench), `metaculus_cup`.
- Dry-run: set `publish_reports_to_metaculus=False`.
- Models configured via `llms={...}` dict (litellm names, e.g.
  `openrouter/anthropic/claude-sonnet-5`).
- GitHub Actions workflows in `.github/workflows/` run main.py on schedule.

## Local environment gotchas (Windows)

- Run on **Python 3.11** (`poetry env use C:\Users\usmb9\AppData\Local\Programs\Python\Python311\python.exe`).
  forecasting-tools applies `nest_asyncio` on import, which breaks litellm's
  HTTP transports on Python 3.14. GitHub Actions workflows also pin 3.11.
- Set `PYTHONUTF8=1` before running — the bot prints emoji banners that crash
  on cp1252 consoles.
- Poetry lives at `C:\Users\usmb9\.local\bin\poetry.exe` (prepend to PATH).

## Keys needed (put in .env, never commit)

- `METACULUS_TOKEN` — create bot account at https://www.metaculus.com/futureeval/participate/
- `OPENROUTER_API_KEY` — https://openrouter.ai/keys (free tournament credits:
  https://forms.gle/aQdYMq9Pisrf1v7d8)
- Optional research: AskNews / Perplexity / Exa keys

## Reference links

- Framework: https://github.com/Metaculus/forecasting-tools
- Template: https://github.com/Metaculus/metac-bot-template
- What other bots do: https://www.metaculus.com/notebooks/43497/what-are-other-bots-doing/
- ForecastBench: https://www.forecastbench.org/
- ACX overview: https://www.astralcodexten.com/p/the-ai-superforecasters-are-here
