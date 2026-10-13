# How to label

You're creating the **gold set**: human labels that the LLM tone scorer and the baselines are
tested against. It's the only ground truth in the project. A model's own scores can't show whether
the model is right, and comparing two LLMs only shows whether they agree.

The labels test the **tone scorer**: are the scores it gives a passage right? They don't test
retrieval, which has its own eval.

## What you'll label, and how long it takes

| Sheet | Items | What you score | Time |
|---|---|---|---|
| Practice (`practice.csv`) | 10 passages | off topic + 5 dimensions | 30–45 minutes, plus 15 comparing with the key |
| Gold tone profiles | about 30 more passages | off topic + 5 dimensions | about an hour |
| Gold polarity | about 50 sentences | positive, negative or neutral | about 20 minutes |
| Re-check, a week later | 10 passages you already labeled | off topic + 5 dimensions | about 15 minutes |

About 1.5–2 hours in total. Your practice labels count as the first 10 gold items, as long as you
label them before opening the key.

The practice sheet is ready now. The other sheets come after the full S&P 500 embed, so the
passages cover every sector.

## Before you start

1. Read `guidelines.md` in full, about 15 minutes. Keep it open while you label.
2. Open the sheet in Numbers, Excel or Google Sheets. Widen the `passage` column and turn on text
   wrapping so you can read each passage.

## Labeling a sheet

**Only fill in these columns:** `off_topic`, `polarity`, `magnitude`, `realization`,
`uncertainty`, `specificity` and `notes`. Don't edit the IDs or the passage text.

For each item:

1. Read the topic, then the whole passage, once.
2. Decide `off_topic` (yes or no). If yes, leave that item's scores empty and move on.
3. Score the five dimensions in column order: `polarity`, `magnitude`, `realization`,
   `uncertainty`, `specificity`. For each one, find the words in the passage that justify a score,
   and pick the score whose definition fits them best. Ask each dimension's own question, and
   check its "Not" rule in the guidelines. The dimensions often move together, but each asks
   something different, and a high score on one shouldn't pull up the next.
4. If you're torn between two scores, pick your best guess and write why in `notes`. Never leave a
   score empty on an on-topic item.

Watch **specificity** in particular. The LLM gave the top score to 69% of passages, and its scores
barely track how many concrete details a passage contains. A 2 needs concrete details *about the
topic*; other details in the same passage don't count.

Save as CSV (UTF-8) with the same file name.

## Working blind

The gold sheets won't show the LLM's scores, and you shouldn't look them up. If you saw them
first, your labels would lean toward them. Then the halo check, which compares the LLM's
correlations between dimensions with yours, would be comparing the LLM with itself.

After you finish a sheet:

1. I merge your labels with the LLM's and list the items where you disagree.
2. You look at only those items. For each, keep your label or change it, with a one-line note
   either way.
3. Both versions are kept. Your **blind** labels feed the halo check and the agreement numbers.
   The **adjudicated** labels are the final gold set that scorers are tested against.

## After the practice sheet

Save your labels first. Then compare them with `practice_key.csv`, which gives my scores and a
reason for each. The key isn't gold either; it's one careful reading.

- **Within one point on most items:** you've understood the rules.
- **Off by two or more, or a different `off_topic`:** reread that dimension's rules. If you still
  disagree with the key, the rule is probably unclear. Note it, and we'll sharpen the guidelines
  before the gold sheets. If a rule changes, re-score just that dimension on the practice items.
  After that the rules are frozen.

## Consistency

- Keep sessions to about 45 minutes; labels drift when you're tired.
- About a week after the gold set, you'll get a re-check sheet with 10 passages you've already
  labeled, shuffled, without your earlier answers. Comparing the two rounds shows how consistent
  one labeler is with themselves.
- If someone else can label about 20 of the same passages, we can also report agreement between
  two people (weighted Cohen's kappa). Reviewers will ask for it.

## What the labels are used for

- **Agreement** between you and each scorer, for each dimension (weighted kappa). A dimension where
  people and LLMs can't agree (kappa below 0.6) gets dropped.
- **Halo check:** if the LLM's dimensions correlate much more than yours do, the LLM's overall
  impression is leaking between scores.
- **Merge check:** if two dimensions correlate above 0.8 in *your* labels, they're probably one
  thing, and get merged.
