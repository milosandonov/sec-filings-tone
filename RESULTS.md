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
in the expected direction in **10 of 15** tests (**6 of 15** when passages about AI, China and tariffs
must name the topic; see the limitations), while FinBERT and the Loughran-McDonald dictionary,
on the same passages, did in **0 of 9**. Polarity alone is too thin for filings: 86% of on-topic
passages score −1. Realization moved more than polarity in 4 of the 6 events. A quarterly follow-up
on Financials' 10-Qs ($1.45) did **not** detect SVB or the rate hikes in its planned before-after
tests (0 of 5), for reasons checked by hand: bank MD&A text already sits at the top of the
realization and specificity scales, and the top 3 passages often miss the event text. Its
quarter-by-quarter charts did show banks' tone on deposits turning from clearly positive in 2021 to
neutral in 2023. Across 12 topics, FY2025 10-Ks describe more risks as already realized than FY2019
ones (realization up for 8 or 9 of them), and AI went from 8% of risk-factor sections to 93%. One
weakness found on the way: the model calls many passages on topic that never name the topic (over
half of the early AI ones), so a word-checked version is kept beside every affected result. About
80% of a 10-K's risk factors and 62% of its MD&A are copied from the year before (FY2025); copied text
scores as specific as new text, so specificity measures concreteness, not novelty, but new text is
clearly more realized.

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

**10 of 15 as expected.** **Word-checked version** (added 2026-10-17 after the results, both kept;
`events.py --named`, `data/tone/events_named.csv`): when passages about AI, China and tariffs count
only if they name the topic, **6 of 15** remain. Lost: export-control polarity (−0.12 [−0.32, +0.05]),
both AI results (below) and tariff uncertainty (+0.04 [−0.02, +0.11]), the four weakest or smallest
tests. The other events have no word list, so their results are unchanged. The planned comparison with unaffected sectors is weaker (2 of 11: rate
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
- **Robustness check, added 2026-10-17 after the results, both versions kept:** before FY2023, 53–59%
  of the passages the model called on topic for AI contain no AI word (limitations). Counting only
  passages that name AI (artificial intelligence, machine learning, "AI", generative and the like),
  9 IT and Communication Services companies remain: polarity +0.30 [0.00, +0.59] and specificity
  +0.22 [−0.22, +0.67], so **neither is detected**; magnitude +0.52 [0.00, +1.00]. Polarity's
  difference from other sectors becomes positive, +0.72 [+0.22, +1.37], on 7 other companies. Part of
  the reported change came from comparing general technology text in FY2021 with AI text in FY2025.

## 4. The LLM profile against FinBERT and Loughran-McDonald

Each baseline was tested against the expected direction of the dimension it stands in for.

| | Expected changes detected |
|---|---|
| LLM tone profile | **10 of 15** (6 of 15 word-checked) |
| FinBERT polarity, LM polarity, LM uncertainty, details per 100 words | **0 of 9** (0 of 9 word-checked) |

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
companies, and the share of companies discussing the topic (10-Q MD&A only, no Q4). The charts were
drawn and then removed when B5 was parked (2026-10-17); the numbers below are from them. What they
show was **not tested in advance**, so read it as description:

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

**Parked (2026-10-17):** SVB may be the wrong event for a quarterly test, so B5 stops here. The scores
(`data/tone/q_*`) and the `--quarterly` code are kept, so everything above can be rerun.

## 7. Topic tone over time and across sectors (annual 10-Ks)

**What the charts show** ([`charts.py`](src/sec_filings_tone/charts.py), PNGs and their numbers in
[`figures/`](figures/)), FY2019–FY2025, 464–494 companies a year. FY2026 is left out: only the 80
companies whose fiscal year doesn't end in December had filed.

- **Line charts** for AI, tariffs and interest rates
  ([`lines_ai.png`](figures/lines_ai.png), [`lines_tariffs.png`](figures/lines_tariffs.png),
  [`lines_interest_rates.png`](figures/lines_interest_rates.png)): per fiscal year, the mean over
  companies of each dimension, risk factors and MD&A drawn separately, with 95% bootstrap
  intervals, and the share of companies with an on-topic passage from each section.
- **Heatmaps**, one per topic (`heatmap_<topic>.png`, all but deposits, which covers Financials only):
  each dimension's mean per sector and fiscal year over the companies discussing the topic. Each
  dimension has one color scale for every topic; cells where fewer than 5 companies discuss the topic
  are hatched.

**How to read the section split:** a company's top 3 passages per topic come mostly from risk
factors, so the MD&A line covers fewer companies (for AI, 85 in FY2025 against 466). Its share
measures how often an MD&A passage made the top 3, not how often MD&A mentions the topic.

None of this was tested in advance; it is description, with the surprising patterns checked by
reading passages.

**AI arrived as a risk everywhere and as good news in results.** Counting only passages that name
AI (the word check; the model alone also counts general technology text as AI, see the limitations),
the share of 10-Ks with an AI passage in risk factors went from 8% (FY2019) and 14% (FY2022) to 93%
(FY2025); the model-only version says 23%, 33% and 94%. Among the companies whose risk factors name AI,
tone was already negative and hedged before ChatGPT and stayed so (polarity −0.88 → −0.93, uncertainty
1.96 → 2.05, FY2019 → FY2025); magnitude (0.60 → 0.86 from FY2022) and specificity (1.10 → 1.32) rose
moderately, realization a little (0.46 → 0.57). So what changed is mainly how many companies treat AI
as a risk. The model-only version shows much larger rises in realization, uncertainty and
specificity, but they come mostly from the non-AI passages it counts in the early years. In MD&A,
AI stayed positive (+0.90, +0.68, +0.88 in FY2019, FY2022, FY2025, on 20–76 companies) with larger
effects (magnitude 0.96 → 1.28) and more uncertainty (0.43 → 1.01). Charts: `lines_ai.png` and the
word-checked `lines_ai_named.png`; the heatmaps show the risk factors spreading to every sector from
FY2023 on in both.

**Tariffs were realized twice** (the word-checked version is nearly identical). In MD&A, tariff text was largely realized in FY2019 (0.94 [0.73,
1.14], after the 2018–19 China tariffs), faded toward hypothetical by FY2023 (0.57), then jumped in
FY2025 (1.18 [1.08, 1.29]), with magnitude 0.57 → 0.99 and the share of 10-Ks with a tariff passage in
MD&A 15% → 42%. Realization in risk factors rose too (0.45 → 0.70). By sector, realization rose most
from FY2024 to FY2025 in Health Care (+0.52), Consumer Staples (+0.47) and Materials (+0.46), least in
Financials (+0.10): the 2025 tariffs reached well beyond the sectors `events.csv` lists, as section 3
found.

**Interest rates: the hikes registered in risk factors.** Risk-factor realization rose from 0.37
(FY2021) to 0.56 (FY2022) and 0.60 (FY2023), then eased (0.49 in FY2025). In MD&A, polarity went from
−0.45 (FY2020, near-zero rates) to −0.18 (FY2023) and back to −0.33, and magnitude peaked in FY2023
(1.18). **One surprise, explained:** risk-factor specificity fell steadily, 1.43 → 1.24. It isn't
vaguer writing. Rate risk factors that name LIBOR or SOFR score 2.0 in every year, the rest about 1.2
in every year, and the LIBOR share fell from 28% to 3% with LIBOR's phase-out (June 2023). Specificity
did what the rules ask: a named benchmark counts as concrete, and when it went, the scores fell.

**Across all 12 topics** (whole 10-K, FY2019 → FY2025, differences of means; their standard errors are
0.01–0.08, largest for AI, so changes below about 0.1 may be noise):

| Topic | Discussed | Polarity | Magnitude | Realization | Uncertainty | Specificity |
|---|---|---|---|---|---|---|
| Geopolitics and war | 92% → 97% | −0.03 | +0.18 | +0.33 | +0.03 | +0.12 |
| Cybersecurity | 100% → 100% | −0.04 | +0.08 | +0.32 | −0.01 | +0.09 |
| Tariffs | 76% → 92% | −0.05 | +0.26 | +0.29 | +0.08 | +0.08 |
| Supply chain | 93% → 97% | −0.12 | +0.15 | +0.26 | +0.12 | −0.07 |
| Inflation and costs | 100% → 100% | −0.13 | +0.13 | +0.21 | +0.21 | −0.01 |
| Labor and workforce | 98% → 99% | −0.06 | +0.08 | +0.19 | +0.08 | +0.01 |
| AI | 28% → 95% | −0.33 | +0.34 | +0.16 | +0.54 | +0.44 |
| Energy transition and climate | 73% → 95% | −0.06 | +0.10 | +0.14 | +0.02 | +0.20 |
| Demand | 100% → 100% | −0.04 | +0.05 | +0.13 | +0.02 | +0.04 |
| Interest rates | 96% → 99% | −0.13 | +0.02 | +0.01 | −0.01 | −0.15 |
| China | 63% → 68% | −0.04 | +0.10 | 0.00 | +0.02 | +0.09 |
| Government policy | 100% → 100% | −0.12 | +0.01 | −0.05 | +0.12 | −0.01 |
| *AI, word-checked* | 12% → 94% | −0.51 | +0.12 | −0.16 | +0.51 | +0.07 |
| *China, word-checked* | 50% → 56% | −0.06 | +0.02 | −0.12 | +0.05 | +0.03 |
| *Tariffs, word-checked* | 68% → 90% | −0.03 | +0.25 | +0.30 | +0.06 | +0.06 |

- **Realization rose for 9 of 12 topics** (each by 0.13 or more, over three standard errors; 8 with
  the word check, under which AI and China fall), **polarity fell for all 12** (beyond two standard
  errors for 10), and magnitude rose for most. Word-checked AI changes mix sections: AI-naming
  passages were more often upbeat MD&A in FY2019 and mostly risk factors by FY2025. 10-Ks in 2025 describe more of their risks as things that have happened, with larger effects.
  The same model scored every year, so this is not model drift, though it may partly be a general
  change in how 10-Ks are written rather than in the topics themselves.
- **Cybersecurity, checked by reading:** the share of passages scored realization 2 went from 21% to
  36%, as wording moved from "if any of these systems fail … we could suffer financial loss" to "we
  have experienced cybersecurity threats … and attacks targeting our information technology systems".
  That fits the SEC's guidance against describing incidents that have happened as hypothetical
  (not verified as the cause).
- **Polarity barely separates sectors:** on the shared scale, every polarity heatmap is uniformly
  red, the −1 of section 2.

## 8. Boilerplate: how much of a 10-K is copied from the year before (Phase 8)

**Method** ([`diff.py`](src/sec_filings_tone/diff.py), fixed in PLAN.md before any result): each
sentence of a 10-K's risk factors and MD&A counts as **carried over** if it closely matches (`difflib`
ratio ≥ 0.9) a sentence of the same section in the company's 10-K a year earlier; a passage's
carryover share is the share of its characters in such sentences. FY2020–FY2025, 2,931 10-Ks (FY2019
has no year before). Checked against an all-pairs comparison on four filings: within 0.2 points. Free:
the saved scores are reused, nothing is scored again.

**How much is copied** ([`figures/carryover.png`](figures/carryover.png)), mean over companies:

| | FY2020 | FY2021 | FY2022 | FY2023 | FY2024 | FY2025 |
|---|---|---|---|---|---|---|
| Risk factors | 72% | 77% | 77% | 79% | 81% | 80% |
| MD&A | 52% | 55% | 58% | 61% | 62% | 62% |

FY2020, the COVID year, has the most rewriting; since then a little more is copied every year.
Sectors differ little: risk factors 77–85% in FY2025 (Real Estate the most), MD&A 57–68% (Financials
the most). MD&A carries over less because its numbers change, though a sentence whose only change is
a figure can still pass 0.9.

**Scores of new and copied text.** Of the on-topic scored passages, 8,587 are *new* (at most 20%
copied), 40,552 *carried over* (at least 80%) and 32,433 mixed. Within each company, new minus
carried over:

| | Polarity | Magnitude | Realization | Uncertainty | Specificity |
|---|---|---|---|---|---|
| Risk factors (452 companies) | −0.06 [−0.08, −0.04] | +0.03 (n.s.) | **+0.18** [+0.13, +0.22] | +0.09 [+0.06, +0.11] | +0.01 (n.s.) |
| MD&A (448 companies) | −0.11 [−0.17, −0.04] | **+0.44** [+0.38, +0.49] | **+0.47** [+0.42, +0.53] | −0.15 [−0.22, −0.07] | +0.08 [+0.04, +0.12] |

- **The planned specificity check fails in risk factors** (expected: new text more specific; +0.01
  [−0.03, +0.04]) and passes only weakly in MD&A (+0.08). Across passages, carryover share and
  specificity barely correlate (Spearman −0.04). The word-checked version is the same. **Read by
  hand:** of 8 copied risk-factor passages scored 2, 6 name a law, product or event (the Fair Labor
  Standards Act, the Bayh-Dole Act, the Bank of Japan's March 2024 rate change, IDEXX's analyzers),
  which the rules count as concrete "even if every company in the industry faces it"; 2 look
  over-scored ("U.S. immigration policies", "we single source a number of parts"). So 10-K boilerplate
  is often concrete boilerplate, and specificity, as defined, measures concreteness, not novelty. The
  check can't validate the dimension; the plan's premise that copied text is generic doesn't hold.
- **Realization separates new from copied text**, with no direction set in advance: new risk-factor
  text is more realized by 0.18, new MD&A by 0.47, with larger effects (magnitude +0.44). New text
  reports what happened, the project's starting hypothesis.
- **The LLM agrees a little less with the baselines on copied text:** Spearman with FinBERT polarity
  +0.37 [+0.35, +0.39] on new passages against +0.33 [+0.32, +0.34] on copied ones; with LM polarity
  +0.36 against +0.27. The intervals resample passages, not companies, so they are too narrow.
- **Does tone in new text react more to the events? Can't tell.** Rerun on new passages only, the
  event tests keep 1 to 9 companies each (`data/tone/events_new.csv`): too few, since only a tenth of
  on-topic passages are new and a company needs one in both 10-Ks.
- **Not done:** the planned "what changed" summary per filing, an LLM call; it fits the Q&A layer.

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
- **The off-topic flag is lenient (found 2026-10-17, from a question about the charts).** The gold set
  confirmed that passages the model flags as off topic are off topic; the reverse error was never
  measured. Checked with topic words (`events.TOPIC_WORDS`): of the passages called on topic, those
  not naming the topic are 53–59% for AI in FY2019–FY2022 (15% in FY2023, 3% in FY2025), 20–23% for
  tariffs until FY2023 (7% in FY2025) and 20–25% for China in every year. The model's evidence shows why: it counts general
  technology (data analytics, blockchain, autonomous vehicles) as AI, and general trade restrictions as
  China or tariffs; in one passage its own evidence says "not import/export tariffs" next to an on-topic
  flag. So AI's early share is overstated and its early tone averages mix in other text. Independent of the
  model, FY2019 10-Ks naming the topic anywhere in risk factors or MD&A: tariffs 78% (the trade war of
  2018–19), China 62%, AI 16%, against the model-based 76%, 63% and 28%: only AI's level is far off.
  A word-checked version of the event tests and of the AI, China and tariff charts is kept next to
  the model's (decision 2026-10-17); the other topics have no clear vocabulary to check.

## Reproduce

```bash
uv run python -m sec_filings_tone.tone retrieve      # passages per cell, free
uv run python -m sec_filings_tone.tone score --budget 30
uv run python -m sec_filings_tone.tone baselines     # FinBERT and LM, local, ~2.6 hours
uv run python -m sec_filings_tone.events             # the event tests, seconds
uv run python -m sec_filings_tone.sentiment gold-report [--adjudicated]
uv run python -m sec_filings_tone.events --named     # word-checked event tests (section 3)
uv run python -m sec_filings_tone.diff carryover     # Phase 8 (section 8), ~2 minutes
uv run python -m sec_filings_tone.diff report        # also the event tests on new text
uv run python -m sec_filings_tone.charts             # figures/ (sections 7 and 8)
# quarterly 10-Qs (section 6, parked): the same commands with --quarterly
uv run python -m sec_filings_tone.tone retrieve --quarterly
uv run python -m sec_filings_tone.tone score --quarterly --budget 3
uv run python -m sec_filings_tone.tone baselines --quarterly   # ~20 minutes
uv run python -m sec_filings_tone.events --quarterly
```

Results are written to `data/tone/` (scores, baselines, event results; the quarterly ones start
with `q_`), which is not in git.
