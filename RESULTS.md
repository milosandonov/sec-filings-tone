# Results: tone scoring (Phases 5–9)

What the tone scorer found, as of 2026-10-18. Retrieval results (Phase 4) are in
[`eval/FINDINGS.md`](eval/FINDINGS.md); the plan, every decision and its date are in [`PLAN.md`](PLAN.md).
Numbers in brackets are 95% bootstrap intervals. "n.s." means the interval includes zero: no
change can be detected, which is not the same as no change.

## Key findings

Reframed 2026-10-17 after an external review; every claim of the earlier one-paragraph summary is
kept, with its caveat. Updated 2026-10-18 for the fiscal-year fix, the one-year event rule and the
AI-boom test retested within MD&A.

- **The data:** Claude Haiku 5.5 scored 126,051 passages from 3,458 S&P 500 10-Ks (FY2019–FY2026) on
  five tone dimensions toward 13 topics for $18.94, and its off-topic flag was right on all 11 gold
  passages it flagged, though it calls on topic many passages that never name the topic (over half
  of the early AI ones), so a word-checked version is kept beside every affected result.
- **Against a person's labels, little is validated:** on one person's blind labels only
  **realization** (has it happened?) passes the agreement rule fixed in advance (weighted kappa 0.67
  on 20 items, bar 0.6), the other four agree only after review toward the model's label, and the
  model isn't detectably better than free word counts on any dimension (closest: realization,
  +0.35 [−0.06, +0.76] in Spearman over a count of realization cues), with polarity impossible to judge.
- **Stability and spread:** scored twice, **magnitude** misses the run-to-run bar fixed in advance
  (kappa 0.78, bar 0.8) but is kept by a dated decision, and polarity barely varies, with 86% of on-topic passages at −1.
- **Events, the planned test:** against sectors the event didn't hit, **0 of 11** expected changes
  are detected (2 of 11 under the old filed rule: two rate-hike differences that are borderline in
  every version, p = 0.04–0.07); the check for a break rather than a trend already under way passes
  for 3 of 13 (COVID realization, export-control polarity and tariff realization, the last failing
  the comparison because the tariffs reached every sector), and no expected change passes both.
- **Events, before-after:** where the event hit, the model moved as expected in **9 of 15** tests
  (10 under the filed rule; word-checked 6 and 6) and FinBERT and the Loughran-McDonald dictionary
  in **0 of 9**, but this count also picks up drift, since from FY2019 to FY2025 realization rose for
  9 of 12 topics and polarity fell for all 12, event or not, and realization beat polarity in 4 of 6
  events before-after but in none of the 4 that have a comparison group.
- **Caveats on the event tests:** "before" is now the last 10-K whose fiscal year ended before the
  event (the owner's decision, 2026-10-17), since the filed rule compared FY2023 with FY2025 for 75%
  of companies in the 2025 tariff test (with one year, the tariffs' realization difference from other
  sectors is +0.00 [−0.12, +0.13]), the AI-boom test, rerun on a top 3 searched within MD&A
  (2026-10-18, $0.85; it was selected on retrieval rank before), finds AI polarity and specificity
  rising in IT and Communication Services MD&A but nearly as much elsewhere (differences +0.11 and
  +0.09, n.s., 38 companies against 40), and the scorer is told each filing's company
  and fiscal year while the baselines aren't, so the tests are free of labels but not of the model's
  knowledge (masked, on 1,903 event-test passages, scores agree with the originals about as well as
  a second run does and realization's changes hold, but COVID's rise in uncertainty falls from +0.21
  to +0.02).
- **Quarterly 10-Qs:** a follow-up on Financials' 10-Qs ($1.45) detected neither SVB nor the rate
  hikes in its planned before-after tests (0 of 5), because bank MD&A already sits at the top of the
  realization and specificity scales and the top 3 passages often miss the event text, though its
  charts show banks' deposit tone turning from clearly positive in 2021 to neutral in 2023.
- **Description:** AI went from 8% to 93% of risk-factor sections (FY2019 → FY2025); 80% of risk
  factors and 62% of MD&A are copied from the year before (FY2025), with new text more realized but
  not more specific than copied text (slightly less in risk factors, within topic and year), so
  specificity measures concreteness, not novelty; and same-sector companies move together only
  negligibly (correlations higher by +0.003 to +0.007), so nearly all of a company's year-to-year
  change is its own, or noise.

## Setup

- **Unit:** a *cell* is one company, one 10-K (fiscal year) and one topic, such as NVDA × FY2023 ×
  China: 42,017 cells. Each cell's passages are hybrid search's top 3 in that 10-K (126,051 passages).
  For AI in MD&A-only analyses, a cell is the top 3 within that 10-K's MD&A instead: 3,436 cells and
  10,308 passages, scored for $0.85 with 1,572 scores reused (2026-10-18, section 3).
  A 10-K's fiscal year is the calendar year its period ends in, except that a 52/53-week year ending
  January 1–7 counts as the year before (fixed 2026-10-17 for 13 companies, PLAN.md Phase 9).
- **Scorer:** `claude-haiku-5-5`, one call per passage, quoting evidence before each score, with the
  rules in [`labels/guidelines.md`](labels/guidelines.md). Message Batches API (half price). Each
  request names the topic, the company, the form, the fiscal year and the section ("Filing: NVDA
  10-K FY2023, Item 1A Risk Factors"), then the passage.
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
  were to carry the evidence for the four not validated by people. On the planned comparison they
  carry little (section 3, reframed 2026-10-17), so those four remain unvalidated. By the rule fixed
  in advance (kappa ≥ 0.6 on blind labels), only realization would be kept.
- **One lead:** the model's magnitude and specificity correlate at +0.71 against the person's +0.10,
  a hint that concrete details raise its magnitude scores, which the rules forbid. 29 items: a lead,
  not a finding.

### The free baselines against the same labels

Added 2026-10-17 (`sentiment.py gold-baselines`, local and free; it reads the saved Haiku scores).
Research question 1 asks how the LLM compares with FinBERT and LM against labels. Spearman
correlation with the person's blind labels, on the 20 gold items both the person and the model call
on topic:

| Person's dimension | Free baseline | Model |
|---|---|---|
| Polarity | FinBERT +0.48 [+0.12, +0.79]; LM +0.22 n.s. | −0.09 n.s. (−1 for 18 of 20) |
| Uncertainty | LM uncertainty share +0.26 n.s.; hedges per 100 words +0.33 n.s. | +0.40 n.s. |
| Specificity | details per 100 words +0.52 [+0.15, +0.80] | +0.54 [+0.15, +0.84] |
| Realization | realization cues per 100 words +0.33 n.s. | **+0.68** [+0.33, +0.91] |

Paired gaps, model minus baseline on the same 20 items (bootstrap over items; a scratch
computation, not in the code): realization +0.35 [−0.06, +0.76], uncertainty against hedges +0.07
[−0.38, +0.51], specificity +0.01 [−0.44, +0.43], and polarity against FinBERT −0.57 [−1.13, +0.01].

- **Answer to question 1, against labels:** on blind labels the model isn't detectably better than
  free word counts on any dimension (realization comes closest), and polarity can't be judged, since
  the model gives nearly every item −1. With 20 items, only gaps of about 0.4 or more could show.
- **On the adjudicated labels the model leads clearly** (realization +0.90 against +0.50,
  uncertainty +0.90 against +0.54 for hedges, specificity +1.00 against +0.61), but that's an upper
  bound: 67 of the 85 disagreements were settled by adopting the model's label.

## 2. Scoring the S&P 500

- All 126,051 passages scored, none failed, **$18.94** (a smoke test on one company first, $0.05).
  That includes the fiscal-year fix (2026-10-17): 501 new passages scored and 968 rescored because
  their request named the wrong fiscal year, for $0.21; $18.94 is what the kept scores cost, and
  $19.09 was paid in all.
- **24.8% of passages are off topic**, and 10% of cells have no on-topic passage ("topic not
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
- **Run-to-run stability (B6, 2026-10-17, $0.79):** 5,000 random passages scored a second time with
  the same request (4,999 still match a passage after the fiscal-year fix; recomputed 2026-10-18).
  The off-topic flag agrees on 97.6%. On passages both runs call on topic:

  | Dimension | Identical | Weighted kappa |
  |---|---|---|
  | Polarity | 95% | 0.89 [0.87, 0.91] |
  | Magnitude | 75% | **0.78** [0.76, 0.79] |
  | Realization | 85% | 0.88 [0.87, 0.89] |
  | Uncertainty | 94% | 0.89 [0.87, 0.91] |
  | Specificity | 86% | 0.84 [0.83, 0.85] |

  Every sector is within a few points of these. **Magnitude falls below the 0.8 run-to-run bar that
  Phase 6 fixed in advance**, by which it would be dropped (the semiconductor pilot had passed it);
  the other four clear it. Magnitude is kept by the owner's decision (2026-10-17, after the result): 0.78 is close to the bar, so the 0.8 rule was relaxed for it, and single-company magnitude changes are read with that noise in mind. Polarity's 95% is helped by 86% of passages scoring −1; its kappa still
  holds. For a single cell's year-to-year change, the scorer's own noise is 19% (polarity) to 46%
  (magnitude) of the variance; the rest is which passages are retrieved and real change (section 9).

## 3. Do the scores react to known events?

**Method** ([`src/sec_filings_tone/events.py`](src/sec_filings_tone/events.py), events and expected
directions in [`events.csv`](events.csv), fixed before results): for each company, its last 10-K
whose fiscal year ended before the event against its first 10-K whose fiscal year ends on or after
it (for the AI boom, on or after 2025-01-01), which are consecutive fiscal years for nearly every
company. The score is the mean over the cell's on-topic passages. Reported: the change where the
event hit, the change elsewhere, and the difference between them. **The planned test is the
difference** (difference-in-differences, PLAN.md Phase 7): an event should move tone where it hit
more than where it didn't. The change where the event hit, on its own, also contains any drift that
every filing shares. The break check (A4, fixed before computing, below) asks whether the change is
larger than the one in the year before, so a trend already under way doesn't count.

**Which "before" 10-K (the owner's decision, 2026-10-17, after the results).** Until then "before"
was the last 10-K *filed* before the event. With an "after" whose fiscal year ends on or after the
event, that rule skipped a year wherever a 10-K ends before the event but is filed after it: in the
2025 tariff test, 368 of the 489 companies with both 10-Ks (75%) were compared two fiscal years
apart, 358 of them FY2023 with FY2025, so the change held two years of drift (found 2026-10-17 in the
external review; not intended). The one-year rule compares consecutive fiscal years for all 494
companies in that test. Its trade-off: a December company's FY2024 10-K was filed weeks after the
first tariff orders, so "before" can already react, which biases the change toward zero. The
one-year rule is primary for the main tests, the event studies and the break check; the filed rule
is kept as the robustness version (`events.py --filed`, `data/tone/events_filed*.csv`, below).

**Every test with an expected direction** (one-year rule; reframed 2026-10-17 after an external
review, since the first version led with the "where the event hit" column alone; updated 2026-10-18
after the fiscal-year fix, and for the AI boom's top 3 searched within MD&A, with the earlier version
kept in italics as the record; the counts below use the new one). Companies: those discussing the topic in both 10-Ks, where the event hit
/ elsewhere; for the break check, those whose "after" 10-K is the next fiscal year and that have the
year before.

| Event (companies) | Dimension | Expected | Where the event hit | Elsewhere | Difference (planned test) | Break check |
|---|---|---|---|---|---|---|
| COVID-19 (468 / no comparison) | realization | up | +0.20 [+0.16, +0.24] ✓ | – | – | **+0.29** [+0.04, +0.54] ✓ (21) |
| | uncertainty | up | +0.13 [+0.10, +0.17] ✓ | – | – | −0.24 [−0.43, −0.05] ✗ (21) |
| Chip export controls, semiconductors (19 / 232) | polarity | down | −0.16 [−0.32, −0.01] ✓ | −0.03 n.s. | −0.13 n.s. ✗ | **−0.28** [−0.56, −0.02] ✓ (18) |
| | magnitude | up | +0.11 n.s. ✗ | −0.02 n.s. | +0.12 n.s. ✗ | +0.09 n.s. ✗ (18) |
| | realization | up | −0.01 n.s. ✗ | −0.03 n.s. | +0.02 n.s. ✗ | +0.16 n.s. ✗ (18) |
| | specificity | up | +0.05 n.s. ✗ | −0.01 n.s. | +0.06 n.s. ✗ | +0.07 n.s. ✗ (18) |
| Rate hikes, Financials and Real Estate (103 / 361) | magnitude | up | +0.14 [+0.06, +0.22] ✓ | +0.04 n.s. | +0.09 n.s. ✗ (borderline) | +0.09 n.s. ✗ (101) |
| | realization | up | +0.13 [+0.02, +0.25] ✓ | +0.12 [+0.05, +0.18] | +0.02 n.s. ✗ | +0.15 n.s. ✗ (101) |
| | specificity | up | +0.01 n.s. ✗ | −0.08 [−0.14, −0.02] | +0.10 n.s. ✗ (borderline) | +0.03 n.s. ✗ (101) |
| SVB, banks (75 / no comparison) | magnitude | up | +0.12 [+0.03, +0.21] ✓ | – | – | +0.09 n.s. ✗ (74) |
| | realization | up | +0.02 n.s. ✗ | – | – | −0.01 n.s. ✗ (74) |
| AI boom, IT and Communication Services MD&A, top 3 searched within MD&A (38 / 40) | polarity | up | +0.36 [+0.14, +0.57] ✓ | +0.25 n.s. | +0.11 n.s. ✗ | can't check |
| | specificity | up | +0.56 [+0.25, +0.87] ✓ | +0.47 [+0.24, +0.72] | +0.09 n.s. ✗ | can't check |
| *AI boom, the record: MD&A part of the whole-10-K top 3 (13 / 11), not interpretable as designed* | *polarity* | *up* | *+0.47 [+0.22, +0.73]* | *0.00 n.s.* | *+0.47 n.s.* | *can't check* |
| | *specificity* | *up* | *+0.73 [+0.19, +1.23]* | *+0.41 n.s.* | *+0.32 n.s.* | *can't check* |
| Tariffs 2025, Industrials, Consumer Discretionary, IT (182 / 222) | realization | up | +0.34 [+0.24, +0.44] ✓ | +0.33 [+0.25, +0.42] | +0.00 n.s. ✗ | **+0.27** [+0.13, +0.41] ✓ (163) |
| | uncertainty | up | +0.03 n.s. ✗ | −0.02 n.s. | +0.06 n.s. ✗ | +0.07 n.s. ✗ (163) |

How to read it:

- **The planned test: 0 of 11.** COVID and SVB have no comparison group, so 11 of the 15 tests have
  one, and none passes. The two closest are the rate hikes, borderline in every version: magnitude
  rose +0.09 [−0.005, +0.19] more in Financials and Real Estate than elsewhere (Welch t-test p =
  0.065), and specificity +0.10 [−0.0001, +0.20] more (p = 0.058); over 5 bootstrap seeds of 10,000
  resamples, the lower bounds run from −0.007 to −0.001. The specificity gap comes from the other
  companies' specificity falling (−0.08), as LIBOR disappeared from their rate risk factors (section
  7), not from a rise where the hikes hit (+0.01 n.s.).
- **The break check: 3 of the 13 that can be checked:** COVID realization (+0.29, on the 21 companies
  whose fiscal year ended in early 2020 and so have a year before), export-control polarity (−0.28,
  18 chipmakers) and tariff realization (+0.27, 163 companies). COVID uncertainty's break is
  negative (−0.24): for those 21 companies, uncertainty had already risen in the year before (+0.22,
  event studies below). Tariff uncertainty isn't a break (+0.07 n.s.). The AI boom (a four-year
  window) can't be checked.
- **No expected change passes both,** now trivially. The rate-hike results aren't clear breaks
  either (+0.09 and +0.03, n.s.), and the tariff realization break fails the comparison because the
  "unaffected" sectors rose as much.
- **Before-after where the event hit: 9 of 15** (10 under the filed rule, the count the first
  headline led with). It also picks up drift (below). Of the 9, 3 have no comparison group (COVID, SVB magnitude) and 6 fail it.
- **Word-checked version** (added 2026-10-17 after the results, both kept; `events.py --named`,
  `data/tone/events_named.csv`): when passages about AI, China and tariffs count only if they name
  the topic, the before-after count falls to **6 of 15**. Lost: export-control polarity (−0.14
  [−0.33, +0.05]), AI polarity (+0.09 [−0.17, +0.35]) and AI specificity (+0.17 [−0.07, +0.43]),
  on 23 IT and Communication Services companies. Against other sectors it is **0 of 11**. (With the
  whole-10-K cells it was 7 and 1: AI polarity's difference was +0.76 [+0.27, +1.39] on 8 companies
  against 7, because those 7 others' AI polarity fell; within MD&A it is −0.14 [−0.84, +0.40], 23
  against 11.) The break check keeps COVID and tariff realization (+0.28
  [+0.12, +0.43], 152 companies) and loses export-control polarity (−0.17 n.s.): **2 of 13**. The
  other events have no word list, so their results are unchanged.
- **The filed rule** (the robustness version, recomputed after the fiscal-year fix): before-after
  **10 of 15** (6 word-checked) and against other sectors **2 of 11** (2 word-checked; 3 with the
  whole-10-K AI cells). Where it
  differs from the table above:

  | Event (companies) | Dimension | Where the event hit | Elsewhere | Difference |
  |---|---|---|---|---|
  | COVID-19 (464) | realization | +0.20 [+0.16, +0.24] ✓ | – | – |
  | | uncertainty | +0.15 [+0.12, +0.18] ✓ | – | – |
  | Chip export controls (19 / 233) | polarity | −0.14 [−0.30, −0.01] ✓ | −0.02 n.s. | −0.12 n.s. ✗ |
  | Rate hikes (103 / 361) | magnitude | +0.14 [+0.06, +0.22] ✓ | +0.04 n.s. | **+0.10** [+0.00, +0.20] ✓ |
  | | specificity | +0.01 n.s. ✗ | −0.09 [−0.15, −0.03] | **+0.11** [+0.01, +0.20] ✓ |
  | AI boom, within MD&A (37 / 40) | polarity | +0.31 [+0.07, +0.55] ✓ | +0.25 n.s. | +0.06 n.s. ✗ |
  | | specificity | +0.58 [+0.27, +0.89] ✓ | +0.47 [+0.24, +0.72] | +0.11 n.s. ✗ |
  | *AI boom, whole-10-K record (13 / 11)* | *polarity* | *+0.40 [+0.15, +0.65]* | *0.00 n.s.* | *+0.40 n.s.* |
  | | *specificity* | *+0.58 [+0.08, +1.12]* | *+0.41 n.s.* | *+0.17 n.s.* |
  | Tariffs 2025 (170 / 179) | realization | +0.32 [+0.22, +0.43] ✓ | +0.42 [+0.31, +0.52] | −0.09 n.s. ✗ |
  | | uncertainty | +0.07 [+0.01, +0.14] ✓ | +0.05 n.s. | +0.02 n.s. ✗ |

  The other six rows differ by at most 0.04 and keep their verdicts. The two passes are the rate
  hikes, borderline here too: magnitude +0.10 (p = 0.050; across seeds the lower bound straddles
  zero) and specificity +0.11 (p = 0.037). Before the fiscal-year fix they were +0.11 [+0.01, +0.21]
  and +0.11 [+0.02, +0.21]. In the filed version the tariffs' negative magnitude and specificity
  differences (−0.15 each; −0.02 and −0.00, n.s., under the one-year rule) were the other sectors'
  extra year: their change spanned two fiscal years for 82% of them, against 58% where the tariffs
  hit.

**What limits these tests:**

- **Drift.** Between FY2019 and FY2025, realization rose by 0.13 or more for 9 of 12 topics and
  polarity fell for all 12 (section 7), event or not. A before-after change where an event hit can be
  this drift, which is why the plan compares with unaffected sectors. The rate hikes show the
  problem: the other companies' realization rose too (+0.12 [+0.05, +0.18]), from drift or because
  the hikes reached every company's filings (A4 below), so Financials' and Real Estate's +0.13 is not
  detectably larger (+0.02 n.s.).
- **Tariffs reached every sector.** In the "unaffected" sectors, the share of 10-Ks discussing
  tariffs rose from 78% to 90% (from 64%, over two years, under the filed rule) and realization rose
  +0.33, as much as where the tariffs hit, so the difference can't isolate the effect.
- **Export controls hit a ceiling.** The chipmakers' China text was already realized and concrete
  from the 2019–2020 rules (Huawei, the entity list), so realization and specificity had no room to
  rise; only polarity and uncertainty moved. And only 19 semiconductor companies discuss China in
  both years.
- **The AI boom, retested within MD&A (design fixed in PLAN.md on 2026-10-18 before any retrieval
  or scoring; `tone.py retrieve --section`, $0.85).** The first version wasn't interpretable as
  designed (decision 2026-10-17, after the external review). A cell was hybrid search's top 3 over
  the whole 10-K, so it had an MD&A passage only if MD&A out-ranked the risk factors. As AI risk
  factors spread, the share of AI cells with any retrieved MD&A passage (on topic or not) fell from
  43% (FY2019) to 26% (FY2025). Among cells discussing AI, the share with an on-topic MD&A passage
  fell from 26% to 18%. So the 13 IT and Communication Services companies (of 89 with both 10-Ks)
  whose cell had an on-topic MD&A passage in both years were selected on retrieval rank, not on what
  their MD&A says. Those numbers are kept as the record (the italic rows above,
  `data/tone/events_whole*.csv`).
  - **The new cells:** for AI, each 10-K's MD&A is now searched on its own. Its top 3 passages within
    Item 7 are ranked exactly as before, with the same query vector. As a check, the same code grouped
    by 10-K reproduces all 10,374 whole-10-K AI passages, and 200 cells re-ranked with separate code
    match. That gives 10,308 passages in the 3,436 10-Ks that have an Item 7. Of those, 1,572 had
    already been scored for the same cell, so their scores were copied; the other 8,736 were scored
    for $0.85. Every MD&A-only analysis of AI uses these cells: this test, A4's MD&A-alone tests and
    event study, Phase 8's new-text tests and the AI charts' MD&A line. Risk factors, the whole 10-K
    and the other topics are unchanged.
  - **Who is in the test:** 88 of the 89 IT and Communication Services companies (390 of 392 others)
    have an MD&A in both 10-Ks. Of these, 38 (40 others) discuss AI there in both.
  - **Result: not detected.** In IT and Communication Services MD&A, AI polarity rose +0.36 [+0.14,
    +0.57] and specificity +0.56 [+0.25, +0.87], as expected. But they rose in the other sectors' MD&A
    too: +0.25 [−0.03, +0.54] and +0.47 [+0.24, +0.72]. So the differences, the planned test, are
    +0.11 [−0.27, +0.46] and +0.09 [−0.30, +0.48]. Magnitude (+0.61 against +0.31) and realization
    (+0.40 against +0.28) also rose in both groups.
  - **What did differ:** the share of IT and Communication Services 10-Ks whose MD&A discusses AI rose
    +0.38, against +0.21 elsewhere (difference +0.16 [+0.04, +0.29]; no direction was set for it).
  - **Recomputed with separate code:** every estimate and company count in every version matches.

  In levels, across the
  whole 10-K: the share of 10-Ks discussing AI went from 57% to 99% in IT and Communication Services
  and from 34% to 94% elsewhere, and AI polarity, mostly from risk factors, turned more negative in
  every sector but Utilities (+0.19, 7 companies in FY2021). AI arrived as a risk almost everywhere,
  and as good news in tech's results.
- **The scorer knows the company and the year.** Each request names them ("Filing: NVDA 10-K FY2023,
  Item 1A Risk Factors"; `sentiment.user_prompt`), so the model could score what it knows about
  that period rather than only what the passage says. The event tests are free of labels, not of the
  model's knowledge. FinBERT and LM see only the text. **Checked 2026-10-17, after the results**
  (`tone.py masked`, $0.27): 1,904 passages of the event tests' before and after cells (the 13
  companies whose fiscal-year labels were then being fixed left out) were scored again with the
  company and year removed ("Filing: a 10-K, Item 1A Risk Factors"). After the fix, 1,903 still match
  a passage (one dropped out of the rankings) and every number below is unchanged; the comparison
  keeps the filed rule, under which the sample was drawn. Per passage, the masked scores agree
  with the originals about as well as a second identical run does (weighted kappa 0.80–0.94, against
  0.79–0.91 for the run-to-run check on the same topics; off-topic flag 94.9% the same, against
  96.9%). Pooled over the events, the masked change where the event hit differs from the original
  only for uncertainty (−0.06 [−0.12, −0.02]; realization +0.02 [−0.04, +0.08]). Most of that is
  COVID: on the 28 sampled companies, uncertainty rose +0.21 [+0.06, +0.35] with the year named and
  +0.02 [−0.14, +0.17] masked. The difference sits in the FY2019 "before" passages, which the model
  scored as more certain when told they were from FY2019 (masked, +0.12 a passage; FY2020, +0.02).
  SVB's uncertainty change also shrinks (−0.11 [−0.21, −0.01]), and AI polarity falls from +0.32 to
  +0.22 n.s. (10 companies, whole-10-K cells). Realization's results come from the text (COVID +0.25 → +0.18, tariffs
  +0.37 → +0.36). Two limits: two thirds of the passages name a year, the company or COVID in their
  own text, which masking the header can't hide (on the 623 that name none, agreement is similar,
  kappa 0.73–0.93); and single events are small (10–56 companies), so only the pooled differences
  are firm. The sample's companies are a subset, so its changes differ from the tables above (COVID
  uncertainty +0.21 here, +0.15 over all 464 companies under the filed rule).
- **Robustness check, added 2026-10-17 after the results, both versions kept:** before FY2023, 52–58%
  of the passages the model called on topic for AI contain no AI word (limitations). The word check
  counts only passages that name AI (artificial intelligence, machine learning, "AI", generative and
  the like).
  - **Within MD&A (since 2026-10-18):** 23 IT and Communication Services companies remain, against
    11 others. At most 24 and 17 could, since that many have an MD&A naming AI anywhere in both 10-Ks.
    Polarity +0.09 [−0.17, +0.35] and specificity +0.17 [−0.07, +0.43]: **neither is detected**, even
    before-after. Their differences from other sectors are −0.14 [−0.84, +0.40] and +0.28 [−0.15,
    +0.70]. Magnitude rose +0.43 [+0.17, +0.70]. Under the filed rule (23 / 11), polarity is +0.07
    n.s. and specificity +0.20 n.s.
  - **FinBERT, the only planned test a baseline passes in the main tests:** its polarity difference
    is +0.19 [+0.005, +0.37], with the lower bound at +0.010 to +0.015 over 5 seeds of 10,000
    resamples (Welch p = 0.077). It comes from the 11 other companies' FinBERT polarity falling
    (−0.16), not from a rise where the boom hit (+0.03 n.s.).
  - **With the whole-10-K cells (the record):** 8 IT and Communication Services companies remained
    under the one-year rule. Polarity +0.33 [+0.08, +0.67] was detected and specificity +0.38 [−0.12,
    +0.88] wasn't; magnitude +0.62 [0.00, +1.25]. The difference from 7 other companies was +0.76
    [+0.27, +1.39]. Under the filed rule (9 companies), neither was detected: polarity +0.30 [0.00,
    +0.59], specificity +0.22 [−0.22, +0.67].

  Much of the model-only rise in both versions comes from comparing general technology text in FY2021
  with AI text in FY2025.

### Robustness (A4): every year around each event, breaks, and each section alone

Fixed in PLAN.md before computing (`events.py --robust`); each also word-checked. Since 2026-10-17
all three use the one-year rule's "before" 10-K.

**Event studies** ([`figures/event_study_*.png`](figures/)): each company's change from its "before"
10-K, for every fiscal year around it ("event year" 0 is that 10-K, 1 the next fiscal year), for the
affected companies and the others. The shapes:

- **COVID: a shock that fades.** Realization rose +0.20 in the next 10-K and stayed between +0.14 and
  +0.26 for years; uncertainty peaked at +0.13 to +0.15 in years 1–2 and fell back to +0.06 by year
  6. Only the 21 companies whose fiscal year ended in early 2020 have a year before: their
  realization was flat that year (0.00), but their uncertainty had already risen (+0.22), which is
  why COVID uncertainty fails the break check.
- **SVB: a one-year spike.** Banks' magnitude was flat in the three years before, rose +0.12 in the
  next 10-K and was back to 0 a year later.
- **Rate hikes:** Financials' and Real Estate's magnitude was flat before (−0.06, −0.03), jumped
  +0.14 and eased; the other companies' barely moved (+0.04). Realization rose in both groups alike
  (+0.13 and +0.12, then +0.20 and +0.19): the hikes reached every company's filings. The other
  companies' specificity fell after 2022 (−0.07 to −0.15), the LIBOR phase-out of section 7.
- **Tariffs 2025: flat for five years, then a sharp rise** in both groups alike (realization +0.34
  in the next fiscal year, against changes within 0.08 in the five years before). Most companies'
  fiscal year ends in December, so their "before" 10-K is FY2024, filed weeks after the first orders;
  it was only slightly more realized than FY2023 (+0.04 where the tariffs hit, +0.08 elsewhere). The
  filed rule skipped that 10-K (section 3).
- **Export controls:** the chipmakers' polarity fell −0.16 in the next 10-K after rising +0.09 in the
  year before, which the break check picks up; with 18–19 companies, the years before swing almost as
  much (−0.13 from two years before to one year before).

**Break check** (the change before → after minus the change the year before, so a trend already under
way doesn't count; only companies whose "after" 10-K is the next fiscal year;
`data/tone/event_breaks.csv`, `event_breaks_named.csv`): the results are the last column of the table
above, moved there on 2026-10-17. Three changes are clear breaks: COVID realization, export-control
polarity and tariff realization. The rate-hike and SVB changes point the right way and look like
breaks in the charts, but subtracting a second, noisy change widens the intervals past zero, so this
check can't rule out a trend. The AI boom (a four-year window) can't be checked, and COVID only on
the 21 companies with a year before. (The first version, with the filed rule, found only the tariff
changes, on 67 companies, mostly with fiscal years not ending in December; the one-year rule checks
163, 58% of them December companies.)

- **Not planned, added 2026-10-17 as a check on drift** (rerun 2026-10-18 with the one-year rule):
  the same break check in the tariff window, on six other topics and all companies with a one-year
  step (459–488), finds no realization break in cybersecurity, demand, labor, geopolitics or
  interest rates (all n.s.), and one in supply chain (+0.18 [+0.09, +0.27]), which tariffs also
  touch. So the tariff break isn't a 2025 shift in how every topic is written. The "unaffected"
  companies show a tariff break too (realization +0.28 [+0.13, +0.42], 166 companies), as expected
  if the tariffs reached every sector.

**Each section alone** (`data/tone/events_1A.csv`, `events_7.csv`): the expected changes detected are
7 of 15 in risk factors (6 word-checked) and 6 of 11 in MD&A (4 word-checked, 5 with the whole-10-K
AI cells; chipmakers' China passages almost never come from MD&A), against 9 of 15 (6) for the whole
10-K. Each event shows up
where one would look for it: rate-hike realization in risk factors (+0.27 [+0.12, +0.42], against
+0.09 n.s. in MD&A), SVB's magnitude in MD&A (+0.18 [+0.04, +0.32], against −0.05 n.s. in risk
factors), COVID's realization in risk factors (+0.22) and its uncertainty in MD&A (+0.20), tariffs
in both (realization +0.25 [+0.15, +0.35] in risk factors, 174 companies; +0.59 [+0.30, +0.93] in
MD&A, 27). AI polarity in risk factors fell (−0.23 [−0.38, −0.10]), the opposite of the MD&A
prediction: AI arrived there as a risk. These counts are before-after changes where the event hit,
so they pick up drift too. And each section's cell is the part of the whole-10-K top 3 that comes
from that section, not a search within it, so which companies have an MD&A passage depends on how
their MD&A ranks against their risk factors. The exception, since 2026-10-18, is AI in MD&A, which is
searched within MD&A (the AI boom above).

## 4. The LLM profile against FinBERT and Loughran-McDonald

Each baseline was tested against the expected direction of the dimension it stands in for.
Expected changes detected (reframed 2026-10-17 after an external review, since the first version
showed only the first column; one-year rule, with the filed rule in brackets):

| | Before-after where the event hit (also picks up drift) | Difference from other sectors (the planned test) |
|---|---|---|
| LLM tone profile | **9 of 15** (6 word-checked) [filed: 10 (6)] | **0 of 11** (0 word-checked) [filed: 2 (2)] |
| FinBERT polarity, LM polarity, LM uncertainty, details per 100 words | **0 of 9** (0 word-checked) [filed: 0 (0)] | **0 of 8** (1 word-checked) [filed: 0 (0)] |

Updated 2026-10-18 for the AI boom's top 3 searched within MD&A (section 3). With the whole-10-K AI
cells, the word-checked counts were 7 [6] and 1 [3] for the LLM and 0 and 0 for the baselines.

In the 7 cases testing the same event and dimension, the LLM's before-after change detects 4 (5
under the filed rule) and the baselines none (e.g. COVID: LM uncertainty share −0.0002 [−0.0008,
+0.0003]; AI boom: FinBERT +0.02 [−0.06, +0.11]). Word-checked, the LLM detects 1 of the 7 (COVID
uncertainty; 2 with the whole-10-K AI cells, AI polarity too). Against other sectors (6 of the 7 have
a comparison), the LLM detects none (under the filed rule, rate-hike specificity, which comes from the
comparison group's fall), and the baselines none, except word-checked FinBERT polarity in the AI boom,
+0.19 [+0.005, +0.37] (Welch p = 0.077). That one too comes from the comparison group's fall
(section 3). FinBERT does pick up COVID's negative turn (−0.03 [−0.04, −0.02]), but reads
tariff text as slightly *more* positive after 2025 (+0.05 [+0.02, +0.07]) where the LLM reads it as
more negative (−0.08 [−0.12, −0.04]).

So the LLM's scores move where the baselines' don't, but on the planned test neither scorer detects
anything. Two differences in what they see are part of the gap. The baselines score the whole
passage and the LLM scores the topic, which is the design's point. And the LLM is told the company
and fiscal year while the baselines see only the text, so some of its movement may come from what it
knows about the period (section 3: with both masked, COVID's uncertainty rise mostly goes; the
realization changes hold).

## 5. Does realization move more than polarity?

The Phase 2 hypothesis: after a shock, risk language moves from hypothetical and generic to realized
and specific more than it turns negative. Change where the event hit (before-after, so drift is
included; one-year rule), as a share of each scale's range (polarity spans 4 points, magnitude and
uncertainty 3, realization and specificity 2):

| Event | Polarity | Magnitude | Realization | Uncertainty | Specificity |
|---|---|---|---|---|---|
| COVID-19 | −3% | +6% | **+10%** | +4% | +1% n.s. |
| Chip export controls | **−4%** | +4% n.s. | 0% n.s. | −4% n.s. | +3% n.s. |
| Rate hikes | +1% n.s. | +5% | **+7%** | 0% n.s. | +1% n.s. |
| SVB | +2% n.s. | **+4%** | +1% n.s. | +2% n.s. | +1% n.s. |
| AI boom (38 companies; MD&A searched on its own, section 3) | +9% | +20% | +20% | +10% | **+28%** |
| Tariffs 2025 | −2% | +8% | **+17%** | +1% n.s. | +7% |

Before-after, realization moved more than polarity in 4 of 6 events (COVID, rate hikes, AI,
tariffs); magnitude moved in 5 of 6, specificity in only 2 (the same under the filed rule). **But
this is the comparison that picks up drift** (reframed 2026-10-17 after an external review), and
realization is the dimension that drifted most: it rose for 9 of 12 topics from FY2019 to FY2025,
event or not, while polarity fell for all 12. Against the comparison group, in the 4 events that
have one, realization's difference is n.s. in all 4, and so is everything else but tariffs' polarity
(−2% of the range: the affected sectors turned slightly more negative than the others). Under the
filed rule, rate hikes' magnitude (+3%) and specificity (+5%) were detected, and in tariffs the
affected sectors moved *less* than the others (magnitude −5%, specificity −7%), which came from the
other sectors' extra year. So the before-after changes fit the hypothesis for realization but don't
establish it, and nothing supports it for specificity. COVID's realization (+10%), with no
comparison group, rose in one year and stayed up, and was flat in the year before for the 21
companies that have one (the event study above), which looks more like a shock than a trend. This
range comparison is a rough reading, not a formal test.

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
[`figures/`](figures/)), FY2019–FY2025, 467–497 companies a year. FY2026 is left out: only the 74
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
factors, so the MD&A line covers fewer companies (for AI, with the whole-10-K cells, 85 in FY2025
against 469). Its share measures how often an MD&A passage made the top 3, not how often MD&A
mentions the topic. AI is the exception since 2026-10-18: its MD&A line is each 10-K's top 3
searched within its MD&A (section 3), 229 companies in FY2025, and its share is how often that top 3
has an on-topic passage. The whole-10-K version is kept as the record (`lines_ai_whole.png`,
`lines_ai_named_whole.png`).

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
searched on its own, the share of 10-Ks with an AI-naming passage went from 6% (FY2019) and 7%
(FY2022) to 34% (FY2025), and AI stayed positive, though less so (+0.73, +0.64, +0.60 in FY2019,
FY2022, FY2025, on 28–170 companies). The effects stayed the same size (magnitude 0.96 → 0.94) and
grew more uncertain (0.49 → 0.87). The whole-10-K version, which saw only the MD&As that out-ranked
the risk factors, showed higher polarity (+0.90, +0.68, +0.88, on 20–76 companies) and growing
effects (magnitude 0.96 → 1.28; uncertainty 0.43 → 1.01). So the "larger effects" it showed came
from which MD&As made the top 3. Charts: `lines_ai.png` and the
word-checked `lines_ai_named.png`; the heatmaps show the risk factors spreading to every sector from
FY2023 on in both.

**Tariffs were realized twice** (the word-checked version is nearly identical). In MD&A, tariff text
was largely realized in FY2019 (0.94 [0.73, 1.15], after the 2018–19 China tariffs), faded toward
hypothetical by FY2023 (0.57), then jumped in FY2025 (1.20 [1.08, 1.31]); from FY2024 to FY2025,
magnitude went 0.56 → 0.99 and the share of 10-Ks with a tariff passage in MD&A 16% → 42%.
Realization in risk factors rose too (0.45 → 0.70). By sector, realization rose most
from FY2024 to FY2025 in Health Care (+0.52), Consumer Staples (+0.47) and Materials (+0.46), least in
Financials (+0.10): the 2025 tariffs reached well beyond the sectors `events.csv` lists, as section 3
found.

**Interest rates: the hikes registered in risk factors.** Risk-factor realization rose from 0.37
(FY2021) to 0.56 (FY2022) and 0.61 (FY2023), then eased (0.49 in FY2025). In MD&A, polarity went from
−0.44 (FY2020, near-zero rates) to −0.19 (FY2023) and back to −0.33, and magnitude peaked in FY2023
(1.18). **One surprise, explained:** risk-factor specificity fell steadily, 1.43 → 1.23. It isn't
vaguer writing. Rate risk factors that name LIBOR or SOFR score 2.0 in every year, the rest about 1.2
in every year, and the share naming either fell from 28% to 3% with LIBOR's phase-out (June 2023;
LIBOR alone, to 0% by FY2025). Specificity
did what the rules ask: a named benchmark counts as concrete, and when it went, the scores fell.

**Across all 12 topics** (whole 10-K, FY2019 → FY2025, differences of means; their standard errors are
0.01–0.08, largest for AI, so changes below about 0.1 may be noise):

| Topic | Discussed | Polarity | Magnitude | Realization | Uncertainty | Specificity |
|---|---|---|---|---|---|---|
| Geopolitics and war | 92% → 97% | −0.02 | +0.18 | +0.33 | +0.03 | +0.12 |
| Cybersecurity | 100% → 100% | −0.04 | +0.08 | +0.31 | −0.01 | +0.09 |
| Tariffs | 76% → 92% | −0.05 | +0.27 | +0.30 | +0.08 | +0.09 |
| Supply chain | 93% → 97% | −0.12 | +0.15 | +0.26 | +0.12 | −0.07 |
| Inflation and costs | 100% → 100% | −0.13 | +0.14 | +0.22 | +0.20 | −0.01 |
| Labor and workforce | 99% → 99% | −0.06 | +0.09 | +0.20 | +0.08 | +0.01 |
| AI | 28% → 95% | −0.34 | +0.34 | +0.17 | +0.55 | +0.45 |
| Energy transition and climate | 73% → 95% | −0.06 | +0.10 | +0.14 | +0.02 | +0.20 |
| Demand | 100% → 100% | −0.04 | +0.06 | +0.13 | +0.02 | +0.04 |
| Interest rates | 96% → 99% | −0.12 | +0.02 | +0.01 | −0.01 | −0.15 |
| China | 63% → 68% | −0.04 | +0.10 | +0.01 | +0.02 | +0.09 |
| Government policy | 100% → 100% | −0.12 | +0.01 | −0.05 | +0.12 | −0.01 |
| *AI, word-checked* | 12% → 94% | −0.51 | +0.12 | −0.15 | +0.51 | +0.06 |
| *China, word-checked* | 50% → 57% | −0.06 | +0.02 | −0.12 | +0.05 | +0.02 |
| *Tariffs, word-checked* | 68% → 90% | −0.03 | +0.26 | +0.30 | +0.05 | +0.07 |

- **Realization rose for 9 of 12 topics** (each by 0.13 or more; over three standard errors for 8,
  AI's +0.17 is 2.9, rechecked 2026-10-18; 8 with the word check, under which AI and China fall),
  **polarity fell for all 12** (beyond two standard errors for 10), and magnitude rose for all 12
  (beyond two standard errors for 8). Word-checked AI changes mix sections: AI-naming
  passages were more often upbeat MD&A in FY2019 and mostly risk factors by FY2025. 10-Ks in 2025 describe more of their risks as things that have happened, with larger effects.
  The same model scored every year, so this is not a change of scorer, though it may partly be a
  general change in how 10-Ks are written rather than in the topics themselves. And each request
  names the fiscal year, so the model's knowledge of the period may play a part (section 3: masking
  it leaves realization's event changes intact, but not COVID's uncertainty; the trends here weren't
  rechecked masked). This shared drift is why a before-after change where an event hit isn't enough
  on its own (section 3).
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
carryover share is the share of its characters in such sentences. FY2020–FY2025, 2,887 10-Ks (FY2019
has no year before; recounted 2026-10-18: the first count, 2,931, also took in FY2026 10-Ks). Checked against an all-pairs comparison on four filings: within 0.2 points. Free:
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

**Scores of new and copied text.** Of the on-topic scored passages, 8,616 are *new* (at most 20%
copied), 41,025 *carried over* (at least 80%) and 32,737 mixed. Within each company, new minus
carried over (the planned comparison, pooling the company's topics and years):

| | Polarity | Magnitude | Realization | Uncertainty | Specificity |
|---|---|---|---|---|---|
| Risk factors (451 companies) | −0.07 [−0.09, −0.05] | +0.03 (n.s.) | **+0.17** [+0.12, +0.22] | +0.09 [+0.07, +0.12] | +0.01 (n.s.) |
| MD&A (451 companies) | −0.11 [−0.17, −0.04] | **+0.44** [+0.38, +0.50] | **+0.47** [+0.41, +0.52] | −0.15 [−0.23, −0.07] | +0.08 [+0.04, +0.12] |

**Within the same topic and year** (added 2026-10-17 after an external review, which recomputed it;
both versions kept). Pooling a company's passages also compares topics and years that have more new
text (supply chain in FY2020, geopolitics and inflation in FY2022, tariffs in FY2025) with those
that have less. Here new and carried-over passages are compared within each company × topic ×
fiscal year, then averaged per company. Only companies with both kinds in the same topic and year
count, so fewer of them:

| | Polarity | Magnitude | Realization | Uncertainty | Specificity |
|---|---|---|---|---|---|
| Risk factors (389 companies) | −0.08 [−0.11, −0.05] | −0.02 n.s. | **+0.17** [+0.09, +0.25] | +0.10 [+0.06, +0.14] | **−0.07** [−0.13, −0.01] |
| MD&A (196 companies) | +0.14 [+0.01, +0.27] | **+0.40** [+0.30, +0.49] | **+0.46** [+0.32, +0.60] | −0.42 [−0.58, −0.26] | +0.07 n.s. |

Realization holds in both sections. Specificity now fails in both, and in risk factors new text is
slightly *less* specific than copied text (word-checked the same, −0.07 [−0.13, −0.01]). MD&A
polarity and uncertainty change sign or size (−0.11 → +0.14; −0.15 → −0.42), so the pooled MD&A
numbers for them partly reflect which topics and years have new text. Within topic and year, new
MD&A text is slightly more positive than copied text (+0.13 n.s. before the fiscal-year fix, +0.14
[+0.01, +0.27] after it, so read it as borderline).

- **The planned specificity check fails in risk factors** (expected: new text more specific; +0.01
  [−0.02, +0.04]) and passes only weakly in MD&A (+0.08); within the same topic and year it fails in
  both (above). Across passages, carryover share and
  specificity barely correlate (Spearman −0.04). The word-checked version is the same. **Read by
  hand:** of 8 copied risk-factor passages scored 2, 6 name a law, product or event (the Fair Labor
  Standards Act, the Bayh-Dole Act, the Bank of Japan's March 2024 rate change, IDEXX's analyzers),
  which the rules count as concrete "even if every company in the industry faces it"; 2 look
  over-scored ("U.S. immigration policies", "we single source a number of parts"). So 10-K boilerplate
  is often concrete boilerplate, and specificity, as defined, measures concreteness, not novelty. The
  check can't validate the dimension; the plan's premise that copied text is generic doesn't hold.
- **Realization separates new from copied text**, with no direction set in advance: new risk-factor
  text is more realized by 0.17, new MD&A by 0.47, with larger effects (magnitude +0.44); within the
  same topic and year, by 0.17 and 0.46 (magnitude +0.40 in MD&A). New text reports what happened,
  the project's starting hypothesis.
- **The LLM agrees a little less with the baselines on copied text:** Spearman with FinBERT polarity
  +0.37 [+0.35, +0.39] on new passages against +0.33 [+0.32, +0.34] on copied ones; with LM polarity
  +0.36 against +0.27. The intervals resample passages, not companies, so they are too narrow.
- **Does tone in new text react more to the events? Can't tell.** Rerun on new passages only, the
  event tests keep 1 to 14 companies each (`data/tone/events_new.csv`): too few, since only a tenth of
  on-topic passages are new and a company needs one in both 10-Ks.
- **Not done:** the planned "what changed" summary per filing, an LLM call; it fits the Q&A layer.

## 9. Do companies in the same sector move together? (research question 4)

**Method** ([`comove.py`](src/sec_filings_tone/comove.py), fixed in PLAN.md before computing): for
each dimension, each company's change in each topic's cell from one fiscal year to the next (both
years discussed, FY2019–FY2025, the 12 topics scored for every company), minus that topic-year's
mean change over all companies, so the shift everyone shared doesn't count. Every pair of companies
is correlated over the topic-years both have (at least 10; typically 55). Test: the mean correlation
of same-sector pairs minus that of different-sector pairs; p-value from 2,000 shuffles of the sector
labels, one-sided, since the hypothesis is that same-sector pairs move together more. The plan
treated it as a hypothesis to test, not a result to expect (wording fixed 2026-10-17: it said "no
direction was expected", which a one-sided test contradicts).

| Dimension | Same sector | Different sectors | Difference | p |
|---|---|---|---|---|
| Polarity | +0.004 | −0.002 | +0.006 | 0.002 |
| Magnitude | +0.001 | −0.003 | +0.004 | 0.005 |
| Realization | +0.003 | −0.003 | +0.006 | 0.001 |
| Uncertainty | +0.000 | −0.003 | +0.003 | 0.03 |
| Specificity | +0.004 | −0.003 | +0.007 | 0.0005 |

121,314 pairs, 13,609 of them in the same sector (rerun 2026-10-17 after the fiscal-year fix).

- **Same-sector pairs do move together more, on every dimension, but by almost nothing:** both
  averages are within 0.004 of zero. The difference is detectable only because there are so many
  pairs. Recomputed for 2,938 random pairs with separate code: identical.
- **Versions:** without removing the shared shift, correlations are 0.01–0.04 and the same-sector
  excess is 0.003–0.008 (p ≤ 0.03). Word-checked: the same, but uncertainty drops out (p = 0.09). On
  text that isn't mostly copied, only polarity stays (+0.008, p = 0.01; the others p = 0.13–0.81), but
  that version has a year less and a fifth fewer pairs, so it can't tell copied boilerplate from lost
  power. New text only: 85 pairs, too few.
- **By sector** (mean same-sector correlation): Consumer Staples, Utilities and Energy move together
  most (up to +0.029, on polarity, magnitude, realization or specificity); Materials least.
- **Why so small (description, not planned in advance):** the shift all companies shared in a
  topic-year explains only 1.3–3.7% of the variance of the changes; adding sector explains another
  2.8–3.2%, against 2.4–2.6% with shuffled sectors, so about half a percentage point is sector-wide.
  The other ~95% is each company's own movement or measurement noise: a cell is 3 passages, and which
  3 can change from year to year. That is why the averaged charts and event tests, over hundreds of
  companies, show clear movement while single pairs barely correlate. The test can't separate a
  weak sector effect from noisy cells. The run-to-run check (section 2) puts the scorer's share of the
  variance of these yearly changes at 19–46% by dimension; the rest is passage selection and real change.

## Limitations

Added 2026-10-17 after an external review: the first six points below.

- **Before-after changes include drift.** Realization rose for 9 of 12 topics and polarity fell for
  all 12 from FY2019 to FY2025, event or not. The 9 of 15 before-after count (10 under the filed
  rule) can't tell an event from that drift; only the comparison with unaffected sectors and the break check can, and they
  find little.
- **Few clean tests.** The planned comparison detects 0 of 11 expected changes (2 of 11 under the
  filed rule, both borderline rate-hike differences) and the break check 3 of 13; none passes both.
  COVID and SVB have no comparison group, the tariffs reached every sector, and the rate-hike
  specificity gap comes from the comparison group's fall.
- **Which "before" 10-K matters.** The filed rule ("before" = last 10-K filed before the date)
  skipped FY2024 for 75% of companies in the 2025 tariff test. The one-year rule, primary since the
  owner's decision of 2026-10-17, compares consecutive fiscal years (all 494 companies in that test),
  but its "before" 10-K can already react to an event that falls between the year's end and the
  filing. Both are reported: under the one-year rule tariff uncertainty is no longer detected and the
  rate-hike differences fall just short.
- **Section-only tests split the whole-10-K top 3.** An AI cell had an MD&A passage only if MD&A
  out-ranked the risk factors, which happened less often as AI risk factors spread, so the first
  AI-boom test was selected on retrieval rank. Since 2026-10-18, AI's MD&A is searched on its own
  ($0.85, section 3). The test is then interpretable, but it detects nothing against other sectors.
  For the other topics, the section-only tests (section 3, A4) still split the whole-10-K top 3.
- **The scorer is told the company and fiscal year,** so the event tests are free of labels but not
  of the model's knowledge of what happened when. The baselines see only the text. Masking both
  (1,904 event-test passages, $0.27; section 3) leaves realization's changes intact but removes most
  of COVID's rise in uncertainty (+0.21 → +0.02), which came from FY2019 passages scored as calmer
  when labeled FY2019.
- **Only realization passes the label rule fixed in advance** (weighted kappa ≥ 0.6 on blind labels),
  and only on its point estimate: 0.67 [0.33, 0.88] on 20 items. One person labeled the gold set,
  with no second rater.
- **Annual data is coarse.** One 10-K before against one after. The quarterly follow-up (section 6)
  covers Financials only and two topics.
- **Small groups.** 19 chipmakers; for the AI boom, 38 IT and Communication Services companies (23
  word-checked), up from 13 (8) with the whole-10-K cells.
- **Short scales hit ceilings.** Realization and specificity have three levels; where a topic was
  already live (China for chipmakers), they can't rise.
- **Changes after seeing results,** each for a design flaw found on inspection and kept alongside the
  first version (`data/tone/events_v1.csv`): "after" became the first 10-K whose fiscal year ends on
  or after the event (the first rule counted 10-Ks filed weeks after the 2025 tariff orders, and one
  filed on the export-control date); export controls narrowed from all of IT to semiconductors (as
  `events.csv`'s note already said); the AI boom's "after" year moved from FY2022 to FY2025. Of 36
  verdicts, the first change altered 3, none on an expected direction. The first change, combined
  with the filed "before", is what made the 2025 tariff test skip FY2024; the one-year rule
  (2026-10-17, also after the results) removes that.
- **The adjudicated labels lean toward the model,** so they can't validate it.
- **Magnitude is the least stable dimension:** scored twice, the same passage gets the same magnitude 75%
  of the time (weighted kappa 0.78, under the 0.8 bar set in Phase 6). Averages over many companies
  absorb that, but single-company magnitude changes are half scorer noise.
- **The off-topic flag is lenient (found 2026-10-17, from a question about the charts).** The gold set
  confirmed that passages the model flags as off topic are off topic; the reverse error was never
  measured. Checked with topic words (`events.TOPIC_WORDS`): of the passages called on topic, those
  not naming the topic are 52–58% for AI in FY2019–FY2022 (15% in FY2023, 3% in FY2025), 20–23% for
  tariffs until FY2023 (6% in FY2025) and 20–26% for China in every year (rechecked 2026-10-18). The model's evidence shows why: it counts general
  technology (data analytics, blockchain, autonomous vehicles) as AI, and general trade restrictions as
  China or tariffs; in one passage its own evidence says "not import/export tariffs" next to an on-topic
  flag. So AI's early share is overstated and its early tone averages mix in other text. Independent of the
  model, FY2019 10-Ks naming the topic anywhere in risk factors or MD&A: tariffs 78% (the trade war of
  2018–19), China 62%, AI 15%, against the model-based 76%, 63% and 28%: only AI's level is far off.
  A word-checked version of the event tests and of the AI, China and tariff charts is kept next to
  the model's (decision 2026-10-17); the other topics have no clear vocabulary to check.

## Reproduce

```bash
uv run python -m sec_filings_tone.tone retrieve      # passages per cell, free
uv run python -m sec_filings_tone.tone score --budget 30
uv run python -m sec_filings_tone.tone baselines     # FinBERT and LM, local, ~2.6 hours
uv run python -m sec_filings_tone.events             # the event tests, seconds
uv run python -m sec_filings_tone.sentiment gold-report [--adjudicated]
uv run python -m sec_filings_tone.sentiment gold-baselines   # FinBERT, LM and counts vs the labels (section 1), local
uv run python -m sec_filings_tone.events --named     # word-checked event tests (section 3)
uv run python -m sec_filings_tone.events --robust    # A4: event studies, breaks, sections
uv run python -m sec_filings_tone.events --filed     # robustness: the filed "before" rule (section 3)
uv run python -m sec_filings_tone.tone retrieve --section          # AI's top 3 within each MD&A (section 3), free
uv run python -m sec_filings_tone.tone score --section --budget 2  # $0.85; copies the scores it already has
uv run python -m sec_filings_tone.tone baselines --section         # ~25 minutes
uv run python -m sec_filings_tone.diff carryover     # Phase 8 (section 8), ~2 minutes
uv run python -m sec_filings_tone.diff report        # also the event tests on new text
uv run python -m sec_filings_tone.comove             # co-movement (section 9), ~1 minute
uv run python -m sec_filings_tone.charts             # figures/ (sections 7 and 8)
# quarterly 10-Qs (section 6, parked): the same commands with --quarterly
uv run python -m sec_filings_tone.tone retrieve --quarterly
uv run python -m sec_filings_tone.tone score --quarterly --budget 3
uv run python -m sec_filings_tone.tone baselines --quarterly   # ~20 minutes
uv run python -m sec_filings_tone.events --quarterly
```

Results are written to `data/tone/` (scores, baselines, event results; the quarterly ones start
with `q_`, the section-level AI ones with `s_`), which is not in git. Run the `--section` commands
before `events`, since every MD&A-only test of AI reads them.
