# Labeling guidelines: tone profile

These rules are used by people labeling the gold set and, word for word, in the prompts for the
LLM scorer and the LLM labelers. Change them only before labeling starts. Once the gold set is
being labeled, they are frozen, so every label means the same thing.

## What you score

Each item is a **passage** (one chunk of a 10-K or 10-Q) and a **topic**. You score how the
passage talks about that topic, from the company's point of view: how the topic affects the
company, as the company describes it. Don't use what you know happened later, or what you think
of the company, or how the stock reacted.

Use only the passage. If it mentions the topic only as one item in a long list and says nothing
else about it, it's still on topic, and most scores will be low.

### Topics

| Topic | Counts as the topic |
|---|---|
| China | Business in or with China: Chinese customers, suppliers and regulators, US export controls aimed at China, US–China trade tensions |
| AI | Artificial intelligence: demand driven by AI, AI products, AI-related competition or regulation |
| Supply chain | Getting inputs and making and shipping products: suppliers, shortages, capacity, logistics, inventory and purchase commitments |
| Demand | Customer demand for the company's products or services, whatever the cause |
| Interest rates | The level and changes of interest rates, and their effect on funding, borrowing, investment income or customers |
| Tariffs | Import and export tariffs and trade investigations that could lead to them (such as Section 232) |
| Deposits and liquidity | Bank deposits, deposit flows and costs, liquidity and funding |
| Government policy | Laws, regulation, subsidies and tax credits that affect the business (CHIPS Act, Inflation Reduction Act, antitrust) |
| Inflation and costs | Rising or falling costs of materials, labor, energy and freight, and price increases in response |
| Geopolitics and war | Wars, sanctions and conflict between countries (Russia and Ukraine, the Middle East). US–China trade and export issues count as China |
| Cybersecurity | Cyberattacks, data breaches, and protecting systems and data |
| Labor and workforce | Hiring, labor shortages, wages, layoffs, strikes and unions |
| Energy transition and climate | Climate change, emissions rules, renewable energy, and clean-energy incentives |

Topics can overlap. A passage on the Inflation Reduction Act's clean-energy credits is on topic for
both government policy and energy transition. Score it separately for each topic it's paired with.

## Step 1: is it on topic?

`off_topic` = **yes** if the passage says nothing about the topic. Then leave the five scores
empty. Otherwise `off_topic` = **no**, and score all five.

## Step 2: the five dimensions

The dimensions sit at three levels. Score each one on its own question, and check the "not"
column before settling on a score: the dimensions often move together, but each asks something
different.

| Level | Dimension | Question |
|---|---|---|
| Evaluative | Polarity | Does the topic help or hurt the company? |
| Evaluative | Magnitude | How big is the effect described? |
| Epistemic | Realization | Has it happened, or is it only possible? |
| Epistemic | Uncertainty | How uncertain does the company say the effects are? |
| Disclosure | Specificity | How concrete is what's said about the topic? |

When a passage mixes several situations (for example, a rule already in force and the risk of
more rules), score **the most advanced stage, the largest stated effect and the most concrete
detail** about the topic. For polarity, score the net direction.

### Polarity (−2 to +2)

| Score | Meaning |
|---|---|
| −2 | The topic clearly hurts the company: losses, charges, restrictions in force, lost sales |
| −1 | A risk of harm, or harm that's mild or partly offset. Most risk factors are −1 |
| 0 | No direction, neutral, or good and bad in balance |
| +1 | Likely helps, or helps mildly |
| +2 | Clearly helps: growth, gains, subsidies received |

**Not:** how big the effect is (that's magnitude). A small, certain loss is −2 on polarity and 1
on magnitude.

### Magnitude (0 to 3)

| Score | Meaning |
|---|---|
| 0 | No effect stated, or only boilerplate ("could materially adversely affect our business") |
| 1 | A minor or limited effect, or the company says it's not material |
| 2 | Significant for a business line, product, region or period |
| 3 | Major for the whole company: a large share of revenue or profit, or company-wide |

**Not:** the direction (a large gain is 3, same as a large loss), and not whether a number is
given (a stated number with no sense of its size doesn't raise magnitude; that's specificity).
"Material" and "significant" in standard risk-factor wording don't count; only a described effect
does.

### Realization (0 to 2)

| Score | Meaning |
|---|---|
| 0 | Hypothetical: something that could happen |
| 1 | Under way or developing: announced, started, being assessed, effects not yet reported |
| 2 | Has happened, and the passage reports effects on the company |

**Not:** how hedged the wording is (that's uncertainty). "New rules took effect in October 2022;
we cannot yet estimate their impact" is realization 1 and uncertainty 3.

### Uncertainty (0 to 3)

| Score | Meaning |
|---|---|
| 0 | The effects are stated as known ("we recorded a $4.5 billion charge") |
| 1 | Mostly known, with an estimate or forecast ("we expect $1.3 billion to $1.5 billion") |
| 2 | A hedged possibility ("may", "could"). The usual score for a hypothetical risk |
| 3 | The company says it can't predict or estimate the effects ("we cannot predict", "remains uncertain", "still assessing") |

**Not:** whether the event happened (that's realization). Uncertainty is about the *effects*.

### Specificity (0 to 2)

| Score | Meaning |
|---|---|
| 0 | Generic: any company in the industry could have written it |
| 1 | Specific to this company or situation, but without concrete details |
| 2 | Concrete details about the topic: named products, counterparties, regulations, dates or amounts |

**Not:** whether it happened. "If H20 exports are restricted, we could lose $X billion of China
revenue" is hypothetical (realization 0) and concrete (specificity 2). Details that aren't about
the topic don't count.

## When you're unsure

Pick your best guess and write a short note. Don't leave a score empty unless the item is off
topic. The notes are how unclear rules get found and fixed.
