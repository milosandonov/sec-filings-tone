# Results: tone scoring (Phases 5–9)

What the tone scorer found, as of 2026-10-17. Retrieval results (Phase 4) are in
[`eval/FINDINGS.md`](eval/FINDINGS.md); the plan, every decision and its date are in [`PLAN.md`](PLAN.md).
Numbers in brackets are 95% bootstrap intervals. "n.s." means the interval includes zero: no
change can be detected, which is not the same as no change.

## In one paragraph

Claude Haiku 5.5 scored 125,583 passages from 3,445 S&P 500 10-Ks (FY2019–FY2026) on five tone
dimensions toward 13 topics, for $18.88. Against one person's labels, it reliably flags passages
that aren't about the topic and agrees on **realization** (has it happened?); the other four
dimensions agree only after the person reviewed the disagreements, mostly by adopting the model's
label. A label-free test gives stronger evidence: around six known events, the model's scores moved
in the expected direction in **10 of 15** tests, while FinBERT and the Loughran-McDonald dictionary,
on the same passages, did in **0 of 9**. Polarity alone is too thin for filings: 86% of on-topic
passages score −1. Realization moved more than polarity in 4 of the 6 events. A quarterly follow-up
on Financials' 10-Qs ($1.45) did **not** detect SVB or the rate hikes in its planned before-after
tests (0 of 5), for reasons checked by hand: bank MD&A text already sits at the top of the
realization and specificity scales, and the top 3 passages often miss the event text. Its
quarter-by-quarter charts do show banks' tone on deposits turning from clearly positive in 2021 to
neutral in 2023.

## Setup

- **Unit:** a *cell* is one company, one 10-K (fiscal year) and one topic, such as NVDA × FY2023 ×
  China: 41,861 cells. Each cell's passages are hybrid search's top 3 in that 10-K (125,583 passages).
- **Scorer:** `claude-haiku-5-5`, one call per passage, quoting evidence before each score, with the
  rules in [`labels/guidelines.md`](labels/guidelines.md). Message Batches API (half price).
- **Dimensions:** polarity (−2 to +2: does the topic help or hurt?), magnitude (0–3: how big is the
  effect?), realization (0–2: has it happened?), uncertainty (0–3: how unpredictable are the
  effects?), specificity (0–2: how concrete?). Plus an off-topic flag.
- **Baselines** on the same passages, scoring the whole passage rather than the topic: FinBERT
  (sentence polarity, averaged), Loughran-McDonald word shares (polarity, uncertainty) and a count of
  numbers, dates and names per 100 words (a rough specificity proxy).

## 1. Does the scorer agree with a person?

40 passages (10 hand-picked practice items, 30 drawn at random the way Phase 9 picks passages),
labeled by one person, blind, then reviewed where they and the model disagreed.

| | Blind labels | After review |
|---|---|---|
| Off topic | model flagged 11, person 0; all 11 are off topic on inspection | 11 of 11 |
| Polarity (kappa) | −0.08 [−0.39, +0.26] | 0.20 |
| Magnitude | 0.12 [−0.10, 0.38] | 0.72 |
| Realization | **0.67** [0.33, 0.88] | 0.90 |
| Uncertainty | 0.52 [−0.02, 0.78] | 0.90 |
| Specificity | 0.50 [0.10, 0.81] | 1.00 |

Weighted kappa on the 20 random items both call on topic; 0.6 was the threshold fixed in advance.

- **What holds:** the off-topic flag, confirmed by hand (none of the 11 passages contains a word for
  its topic), and realization, which passes on blind labels.
- **Why the rest is open:** the blind labels scored what each passage says rather than what it says
  about the topic, which depresses agreement; and in review, 67 of 85 disagreements were settled by
  adopting the model's label, which raises agreement by construction. Neither column is a clean
  measure. Polarity can't be judged at all: the model gives −1 to 18 of 20 items.
- **Decision:** no more labeling (2026-10-16). All five dimensions were scored; the event tests below
  carry the evidence for the four not validated by people.
- **One lead:** the model's magnitude and specificity correlate at +0.71 against the person's +0.10,
  a hint that concrete details raise its magnitude scores, which the rules forbid. 29 items: a lead,
  not a finding.

## 2. Scoring the S&P 500

- All 125,583 passages scored, none failed, **$18.88** (a smoke test on one company first, $0.05).
- **24.7% of passages are off topic**, and 10% of cells have no on-topic passage ("topic not
  discussed", kept apart from neutral tone). Rare topics are off topic most often: AI 62%, China 60%,
  tariffs 53%, against government policy 1% and demand 4%.
- **Score spread** on on-topic passages:

  | Dimension | Distribution |
  |---|---|
  | Polarity | −2: 5%, **−1: 86%**, 0: 5%, +1: 4%, +2: 1% |
  | Magnitude | 0: 27%, 1: 42%, 2: 29%, 3: 2% |
  | Realization | 0: 47%, 1: 22%, 2: 31% |
  | Uncertainty | 0: 9%, 1: 3%, **2: 78%**, 3: 9% |
  | Specificity | 0: 10%, 1: 32%, 2: 58% |

  Polarity barely varies, as the project's premise expects for risk language that is negative by
  design. Realization varies the most.

## 3. Do the scores react to known events?

**Method** ([`src/sec_filings_tone/events.py`](src/sec_filings_tone/events.py), events and expected
directions in [`events.csv`](events.csv), fixed before results): for each company, the last 10-K
filed before the event against the first 10-K whose fiscal year ends on or after it (for the AI
boom, on or after 2025-01-01). The score is the mean over the cell's on-topic passages. Reported:
the change where the event hit, the change elsewhere, and the difference between them.

**Tests with an expected direction** (change where the event hit; n = companies discussing the
topic in both 10-Ks):

| Event | Dimension | Expected | Change | |
|---|---|---|---|---|
| COVID-19 (n = 464) | realization | up | +0.20 [+0.16, +0.25] | ✓ |
| | uncertainty | up | +0.15 [+0.11, +0.18] | ✓ |
| Chip export controls, semiconductors (n = 19) | polarity | down | −0.14 [−0.30, −0.01] | ✓ |
| | magnitude | up | +0.09 (n.s.) | ✗ |
| | realization | up | −0.04 (n.s.) | ✗ |
| | specificity | up | +0.05 (n.s.) | ✗ |
| Rate hikes, Financials and Real Estate (n = 103) | magnitude | up | +0.14 [+0.06, +0.22] | ✓ |
| | realization | up | +0.13 [+0.02, +0.25] | ✓ |
| | specificity | up | +0.01 (n.s.) | ✗ |
| SVB, banks (n = 75) | magnitude | up | +0.12 [+0.03, +0.21] | ✓ |
| | realization | up | +0.02 (n.s.) | ✗ |
| AI boom, IT and Communication Services MD&A (n = 13) | polarity | up | +0.40 [+0.15, +0.65] | ✓ |
| | specificity | up | +0.58 [+0.08, +1.12] | ✓ |
| Tariffs 2025, Industrials, Consumer Discretionary, IT (n = 170) | realization | up | +0.32 [+0.22, +0.43] | ✓ |
| | uncertainty | up | +0.07 [+0.01, +0.14] | ✓ |

**10 of 15 as expected.** The planned comparison with unaffected sectors is weaker (2 of 11: rate
hikes' magnitude +0.11 and specificity +0.11 more than elsewhere), because good comparison groups are
rare:

- **Tariffs reached every sector.** In the "unaffected" sectors, the share of 10-Ks discussing
  tariffs rose from 64% to 90% and realization rose +0.42, so the difference can't isolate the effect.
- **Export controls hit a ceiling.** The chipmakers' China text was already realized and concrete
  from the 2019–2020 rules (Huawei, the entity list), so realization and specificity had no room to
  rise; only polarity and uncertainty moved. And only 19 semiconductor companies discuss China in
  both years.
- **The AI result rests on 13 companies** that discuss AI in MD&A in both FY2021 and FY2025. In levels:
  the share of 10-Ks discussing AI went from 58% to 99% in IT and Communication Services and from 34%
  to 94% elsewhere; across the whole 10-K, mostly risk factors, AI polarity turned more negative in
  every sector. AI arrived as a risk everywhere, and as good news in tech's results.

## 4. The LLM profile against FinBERT and Loughran-McDonald

Each baseline was tested against the expected direction of the dimension it stands in for.

| | Expected changes detected |
|---|---|
| LLM tone profile | **10 of 15** |
| FinBERT polarity, LM polarity, LM uncertainty, details per 100 words | **0 of 9** |

In the 7 cases testing the same event and dimension, the LLM detects 5 and the baselines none
(e.g. COVID: LM uncertainty share −0.0003 [−0.0008, +0.0004]; AI boom: FinBERT +0.08 [−0.06, +0.22]).
FinBERT does pick up COVID's negative turn (−0.03 [−0.04, −0.02]), but reads tariff text as slightly
*more* positive after 2025 (+0.05) where the LLM reads it as more negative (−0.07).

The baselines score the whole passage and the LLM scores the topic. That is the design's point, but
it is also part of the gap.

## 5. Does realization move more than polarity?

The Phase 2 hypothesis: after a shock, risk language moves from hypothetical and generic to realized
and specific more than it turns negative. Change where the event hit, as a share of each scale's
range (polarity spans 4 points, magnitude and uncertainty 3, realization and specificity 2):

| Event | Polarity | Magnitude | Realization | Uncertainty | Specificity |
|---|---|---|---|---|---|
| COVID-19 | −3% | +6% | **+10%** | +5% | +1% n.s. |
| Chip export controls | **−4%** | +3% n.s. | −2% n.s. | −5% | +3% n.s. |
| Rate hikes | +1% n.s. | +5% | **+7%** | 0% n.s. | +1% n.s. |
| SVB | +2% n.s. | **+4%** | +1% n.s. | +2% n.s. | +1% n.s. |
| AI boom (13 companies) | +10% | +24% | +19% | +8% n.s. | **+29%** |
| Tariffs 2025 | −2% | +8% | **+16%** | +2% | +7% |

Realization moved more than polarity in 4 of 6 events (COVID, rate hikes, AI, tariffs); magnitude
moved in 5 of 6. Specificity moved in only 2, so the hypothesis holds for realization, not for
specificity. This range comparison is a rough reading, not a formal test.

## 6. Quarterly 10-Qs: Financials, deposits and interest rates

**What was done** (2026-10-17, PLAN.md option B5): the same scorer on every 10-Q of the 76 Financials
companies, MD&A only, quarters ending 2019-02 to 2026-06, for two topics, deposits and liquidity and
interest rates. A cell is one company, one 10-Q and one topic: 3,400 cells, 10,196 passages (hybrid
search's top 3 in that 10-Q), **$1.45**, none failed. The request is identical to the annual run's,
so the scores compare. There is no Q4: the Q4 report is the 10-K, which isn't comparable because it
includes risk factors.

- **Off topic:** 12.8% of passages (deposits 6.6%, interest rates 18.9%); 1.5% of cells have no
  on-topic passage.
- **Score spread:** unlike the annual table above, because 10-Qs have almost no risk-factor text.
  On-topic passages score polarity +1 for 40% (−1: 30%, 0: 25%), realization 2 for 74%,
  specificity 2 for 85%, uncertainty 0 for 57%. The banks' annual **MD&A** passages on these topics
  score the same way (realization 2: 69%, specificity 2: 99%); their annual risk-factor passages are
  the opposite (polarity −1: 96%).

### Event tests

**Method** (fixed in [`events_quarterly.csv`](events_quarterly.csv) before any quarterly score
existed): each company's last 10-Q filed before the event against its first 10-Q whose quarter ends
on or after it. For calendar-year companies that is Q3 2022 against Q1 2023 for SVB, and Q3 2021
against Q1 2022 for the first rate hike.

| Event | Dimension | Expected | Change | |
|---|---|---|---|---|
| SVB (2023-03-10), banks, deposits and liquidity (n = 13) | magnitude | up | +0.15 [−0.08, +0.38] | ✗ |
| | realization | up | −0.05 [−0.23, +0.15] | ✗ |
| Rate hikes (2022-03-16), all Financials, interest rates (n = 72) | magnitude | up | −0.05 [−0.17, +0.08] | ✗ |
| | realization | up | +0.03 [−0.10, +0.16] | ✗ |
| | specificity | up | −0.05 [−0.14, +0.05] | ✗ |

**0 of 5 as expected.** For SVB, the banks' change minus the other 61 Financials' change is also
zero on every dimension (magnitude +0.13 [−0.14, +0.39], realization +0.02 [−0.20, +0.24]). Polarity
had no expected direction: banks −0.21 [−0.51, +0.10] for SVB, all Financials −0.09 [−0.30, +0.10]
for the rate hikes. The baselines do no better. The one with an expected direction, details per 100
words for rate-hike specificity, fell instead, −0.52 [−0.93, −0.12]. FinBERT polarity changed −0.06
[−0.14, +0.00] for the rate hikes, and LM polarity −0.003 [−0.006, −0.001].

**Why nothing was detected (checked by hand).** The numbers were recomputed with separate code, which
gives the same estimates, and passages were read for KeyCorp, Bank of America, PNC and Travelers.

- **No room at the top.** Bank MD&A reports deposits as concrete, realized facts every quarter
  ("deposits totaled $144.9 billion, a decrease of $7.7 billion": KeyCorp, Q3 2022). Before SVB, the
  banks' deposit cells already averaged realization 1.74 and specificity 1.95 of 2. Deposits were
  already falling in 2022 as rates rose, so Q3 2022 was no calm baseline.
- **The event text lowers realization.** What SVB added is mostly regulation in progress ("Following
  the failures of SVB and Signature … the Federal Reserve created a new Bank Term Funding Program …
  consideration is being given to strengthening …"), which the rules score realization 1 and
  uncertainty 2. That is below the routine balance reports it displaces.
- **Retrieval often misses the event.** All 13 banks' Q1 2023 MD&As mention the bank failures (1 to
  7 passages each, against almost none in Q3 2022). Yet the cell's top 3 includes such a passage for
  only 8 of them, none for Bank of America, Citi, Fifth Third, US Bancorp and Wells Fargo.
- **The same text, quarter after quarter.** 40% of the SVB "after" passages and 32% of the rate-hike
  ones closely match (difflib ratio ≥ 0.8) a passage from the same company's "before" 10-Q. For
  interest rates, retrieval favors the market-risk section, such as PNC's description of its net
  interest income simulation, which is hypothetical (realization 0) in every quarter.
- **The rate-hike window is early.** Q1 2022 ends two weeks after a single 0.25-point hike. A later
  window was proposed before the full run and not adopted; the charts below show the changes
  building through 2022–23 instead.
- **13 banks** is a small group: the SVB magnitude interval spans 0.46 points.

### Charts: tone by quarter

Per calendar quarter, the mean over companies of each cell's tone, with 95% bootstrap intervals over
companies, and the share of companies discussing the topic (10-Q MD&A only, no Q4):
[`figures/q_deposits_and_liquidity.png`](figures/q_deposits_and_liquidity.png),
[`figures/q_interest_rates.png`](figures/q_interest_rates.png),
[`figures/q_deposits_banks_vs_others.png`](figures/q_deposits_banks_vs_others.png) (numbers in the
CSVs beside them). What they show was **not tested in advance**, so read it as description:

- **Banks' polarity on deposits** went from +1.05 [+0.82, +1.28] in Q3 2020 to +0.03 [−0.15, +0.21]
  in Q1 2023 and −0.10 [−0.41, +0.18] in Q3 2023, then back to +0.83 by Q2 2026. The fall began with
  the rate hikes in 2022, not at SVB. On reading: in 2021, 78% of the banks' on-topic deposit passages
  scored +1 or +2 ("Deposit growth was strong … driven by additional government stimulus programs");
  in 2023, 37% scored −1 or −2 ("net outflows of $42.9 billion in money market savings"). Other
  Financials stayed between +0.2 and +0.5 throughout.
- **Banks' uncertainty on deposits** doubled in Q2 2022, from 0.31 to 0.67, and stayed higher.
- **Interest rates, all Financials:** polarity rose from −0.43 [−0.57, −0.29] in Q1 2022 to +0.04
  [−0.16, +0.23] in Q2 2023. Magnitude (1.12 → 1.34) and realization (1.38 → 1.53) rose by Q3 2023,
  with overlapping intervals.
- **Almost every company discusses both topics** in every quarter: 95–100% of Financials for
  interest rates, 97–100% for deposits.

**What this means for the design:** a before-after test on one filing each side suits events that
change the text at once, and those whose effects build over quarters show better as a series. Top 3
passages for a fixed topic query favor sections that recur every quarter, so event text is often
not scored. Possible fixes, to decide before running them: score only text that is new against the
previous filing (Phase 8), or add event-specific queries.

## Limitations

- **Annual data is coarse.** One 10-K before against one after. The quarterly follow-up (section 6)
  covers Financials only and two topics.
- **Small groups.** 19 chipmakers and 13 companies for the AI result.
- **Short scales hit ceilings.** Realization and specificity have three levels; where a topic was
  already live (China for chipmakers), they can't rise.
- **One labeler,** whose blind labels misread the topic rule; no second person.
- **Changes after seeing results,** each for a design flaw found on inspection and kept alongside the
  first version (`data/tone/events_v1.csv`): "after" became the first 10-K whose fiscal year ends on
  or after the event (the first rule counted 10-Ks filed weeks after the 2025 tariff orders, and one
  filed on the export-control date); export controls narrowed from all of IT to semiconductors (as
  `events.csv`'s note already said); the AI boom's "after" year moved from FY2022 to FY2025. Of 36
  verdicts, the first change altered 3, none on an expected direction.
- **The adjudicated labels lean toward the model,** so they can't validate it.

## Reproduce

```bash
uv run python -m sec_filings_tone.tone retrieve      # passages per cell, free
uv run python -m sec_filings_tone.tone score --budget 30
uv run python -m sec_filings_tone.tone baselines     # FinBERT and LM, local, ~2.6 hours
uv run python -m sec_filings_tone.events             # the event tests, seconds
uv run python -m sec_filings_tone.sentiment gold-report [--adjudicated]
# quarterly 10-Qs (section 6): the same commands with --quarterly, then the charts
uv run python -m sec_filings_tone.tone retrieve --quarterly
uv run python -m sec_filings_tone.tone score --quarterly --budget 3
uv run python -m sec_filings_tone.tone baselines --quarterly   # ~20 minutes
uv run python -m sec_filings_tone.events --quarterly
uv run python -m sec_filings_tone.charts                       # figures/
```

Results are written to `data/tone/` (scores, baselines, event results; the quarterly ones start
with `q_`), which is not in git.
