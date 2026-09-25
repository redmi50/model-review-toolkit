# Predictive Model Methodology Review Toolkit

A toolkit that reviews how a predictive model was evaluated by recomputing what the
project reported, rather than by reading what the project said about itself.

A review here is not an opinion. Each check measures a quantity, and the severity of a
finding is derived from how large that quantity turned out to be. A model reported at
0.93 area under the curve whose honest score is 0.46 raises a high severity finding with
the number 0.46 in it; a model reported at 0.94 whose honest score is 0.935 raises
nothing. That is the whole design, and it is what makes the output arguable rather than
persuasive.

## Why this exists

Reading a methodology section tells you what the analyst intended. It does not tell you
what the code did. The gap between the two is where the expensive mistakes live, and it
is invisible from the write-up, because a leaking feature is described as a strong
predictor and a shuffled split over time-ordered rows is described as cross validation.

The failure this toolkit is built to catch is not a bad model. It is a good model with an
optimistic number attached to it, which is worse than a bad model, because the number is
what gets budgeted against.

Three decisions follow from that.

* Severity comes from a measured quantity, never from a phrase. The checks compute an
  optimism gap, the amount by which the reported score overstated the honest one, or a
  share, the proportion of something that should not have it, such as the fraction of test
  rows that already appear in training. Both are mapped through bands stated in the code.
* The reviewer uses the project's own protocol. A holdout stays a holdout and cross
  validation stays cross validation, with the estimator the project chose. The only thing
  a check changes is which columns or which rows it hands in. Otherwise the review would
  be measuring the reviewer's estimator instead of the evaluation.
* A clean project must come back clean. The suite ships a control project that was
  evaluated carefully, and a test asserts it raises nothing. Without that control, eight
  checks that fire on everything would look identical to eight checks that work.

## Quickstart

```bash
git clone <repository-url>
cd model-review-toolkit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# See the fixture projects and the settings that decide which checks apply
python -m model_review list-projects

# See the checks
python -m model_review list-checks

# Review every fixture and write reports/review.md and reports/reviews.json
python -m model_review run

# Review one project
python -m model_review run --projects leaky_feature

# Re-render the markdown from a saved run without recomputing anything
python -m model_review report --reviews reports/reviews.json

# Run the tests
python -m pytest -q
```

`run` writes two artefacts: `review.md` for a reader and `reviews.json` for anything
downstream, which carries the full evidence dictionary for every finding rather than only
the sentence.

## Project layout

```
model-review-toolkit/
  model_review/
    modelling.py    the fixed evaluation protocol: seeds, folds, estimators, the split
    datasets.py     the fixture projects, each with one documented evaluation flaw
    findings.py     a finding, the severity bands, and the review report
    checks.py       the eight checks, each measuring one thing
    report.py       markdown and JSON rendering
    __main__.py     command line interface
  tests/
    test_modelling.py   the shared primitives: the holdout, the folds, the screen
    test_datasets.py    fixture shape, validation, and every stored published figure
    test_checks.py      one class per check, the applicability rules, the bands
    test_report.py      the two report formats and the command line
```

## The protocol

Every number the toolkit produces comes from one place, `model_review/modelling.py`, so a
reviewer can disagree with the protocol in one file rather than in eight checks.

| Constant | Value | What it fixes |
| --- | --- | --- |
| HOLDOUT_TEST_SIZE | 0.30 | The share of rows held out for single split projects. |
| SEED | 0 | Every random split, fold assignment and forest. |
| CV_FOLDS | 5 | The fold count for cross validated projects. |
| FOREST_TREES | 250 | The forest used by the ranking check and the forest fixtures. |
| PERMUTATION_REPEATS | 20 | The shuffles behind each permutation importance estimate. |

The holdout is defined once, as row indices, and the scorer and the structural checks both
consume those indices. That is deliberate: a scorer and an overlap check that each take
their own split can disagree about which rows were held out, and then a report contradicts
itself.

## The fixture projects

Seven projects. Six contain one documented evaluation flaw each, and one is a control.

| Project | Rows | Features | Reported | How it was evaluated |
| --- | --- | --- | --- | --- |
| clean_baseline | 1500 | 5 | roc_auc 0.810 | One stratified holdout, taken before any modelling. |
| leaky_feature | 2000 | 7 | roc_auc 1.000 | Five fold cross validation over a table that includes a closure date. |
| temporal_drift | 1200 | 4 | roc_auc 0.936 | Five fold cross validation over time-ordered rows, shuffled first. |
| selection_outside_cv | 90 | 15 of 3000 | roc_auc 0.928 | The top fifteen candidates chosen on all rows, then cross validated. |
| duplicate_rows | 2120 | 30 | roc_auc 0.997 | Every row replicated five times, then one holdout. |
| imbalanced_metric | 3000 | 5 | accuracy 0.974 | One holdout reporting accuracy where 2.7 percent of rows are positive. |
| importance_method | 900 | 8 | roc_auc 0.823 | One holdout, drivers ranked by the forest's impurity importance. |

Every published figure is stored as a literal rather than computed at import time, and a
test recomputes it from the fixture under the same protocol the checks use. If a fixture
or its stored number drifts, the test suite fails rather than a report quietly becoming
wrong. That is the load bearing test of this repository.

## The checks

Eight checks. The first four compare a reported score against an honest one; the last four
measure a structural property or apply a rule.

| Check | Applies when | What it measures |
| --- | --- | --- |
| reproduce_reported_score | Always | The distance between the published figure and the stated protocol. |
| leakage | Always | The score drop when a column correlated with the label above 0.98 is removed. |
| temporal_evaluation | The split is temporal | The reported score against a forward chaining split. |
| selection_outside_cv | Selection was global | The reported score against the same screen redone inside each fold. |
| row_replication | Always | The share of test rows whose feature vector already appears in training. |
| class_imbalance_metric | Accuracy is reported on a rare class | The base rate, the constant predictor, and the missed positive share. |
| importance_method | Drivers were ranked by impurity | The share of published weight on columns with permutation importance at or below zero. |
| baseline_lift | roc_auc is reported | The shortfall of the reported score below the no skill score plus a margin. |

The severity bands are stated next to the quantity each applies to. For a score gap, a
drop of 0.03 or more is medium, 0.10 or more is high, and a gap at or below zero is not a
finding at all, because a project whose honest score is higher than its reported one was
not being optimistic whatever the method was called.

Reproduction carries a publication tolerance of 0.0005. A figure published to three
decimal places and recomputed to four agree to within half of the last published digit, so
a difference below that is rounding rather than a discrepancy, and the check stays silent.

## Results

Produced by `python -m model_review run` on the committed fixtures. The data is generated
from fixed seeds, so these figures reproduce exactly.

| Project | Findings | High | Weight |
| --- | --- | --- | --- |
| clean_baseline | 0 | 0 | 0 |
| leaky_feature | 1 | 1 | 3 |
| temporal_drift | 1 | 1 | 3 |
| selection_outside_cv | 1 | 1 | 3 |
| duplicate_rows | 1 | 1 | 3 |
| imbalanced_metric | 1 | 1 | 3 |
| importance_method | 1 | 1 | 3 |

Seven projects reviewed, 18 total severity weight, 6 high severity findings.

The specific measured evidence:

| Project | Measured | Reported | Honest |
| --- | --- | --- | --- |
| leaky_feature | roc_auc falls 0.2648 when the closure date is removed | 1.000 | 0.735 |
| temporal_drift | roc_auc falls 0.3435 under forward chaining | 0.936 | 0.593 |
| selection_outside_cv | roc_auc falls 0.4685 when the screen is redone in fold | 0.928 | 0.460 |

Three projects are measured structurally rather than by a score gap, and the reason is in
the code. Row replication barely moves a ranking metric, because a repeated row is still
ranked correctly, so the honest quantity is the share of test rows that were memorised:
98.4 percent of them, 626 of 636. Impurity importance and permutation importance disagree
about which columns matter rather than about the score, so the honest quantity is the
share of published weight on columns that do not improve the held out score: 46.8 percent.
The rare event project scores 0.974 accuracy where a constant predictor scores 0.973 and
95.8 percent of the positive cases are missed, so the model is not the problem and the
metric is.

`clean_baseline` is the one to read first. It scores 0.810 area under the curve on a
balanced problem with a holdout taken before any modelling, and the toolkit raises
nothing, across all eight checks.

## Adding a fixture

1. Write a builder in `model_review/datasets.py` that generates the data from a fixed
   seed, states how it was evaluated, and records the score that evaluation produces.
2. Add it to `BUILDERS`. The id becomes available to `run --projects` and to every
   parametrised test.
3. Run `python -m pytest -q tests/test_datasets.py`. The stored-figure test will fail
   until the literal matches what the stated protocol actually produces, which is the
   point: a fixture cannot ship with a number that is not its own.
4. Run `python -m model_review run --projects <id>` and check that the finding, if any,
   is the one you intended and that its severity follows from the measured size.

## Limitations

The checks look for a specific set of flaws: label leakage through one column, a shuffled
split on ordered rows, selection outside the fold, replicated rows, an uninformative
metric, an unearned importance ranking, and a score on the no skill line. A real
methodology failure outside that list will not be found. The list is the whole
contribution; it is not a substitute for a thorough review.

The leak check looks for a single column that is nearly a copy of the label, so a leak
spread across several columns that jointly reconstruct the outcome is missed. The
threshold of 0.98 is high on purpose, so that a merely strong predictor is not reported as
a defect, at the cost of missing a leak that is only strong rather than near perfect.

Every fixture is synthetic or public and small enough to run in seconds, so the numbers
here are evidence about the checks and about the plumbing rather than about any real
project. The effect sizes on real data will be smaller and noisier, and a medium severity
finding on a real project should be read as a question rather than a verdict.

The checks recompute using the toolkit's own estimators, a scaled logistic regression and
a fixed forest. A project using gradient boosting will not be reproduced exactly, and the
reproduction check will say so. That is a limitation of recomputing someone else's model
rather than re-running their code, and the honest fix is to run their code, which this
toolkit does not do.

The severity bands are judgement calls, stated in full so a reader can disagree with the
threshold rather than guess at it. Moving HIGH_GAP from 0.10 to 0.05 would reclassify
real findings, and nothing in the toolkit would notice.
