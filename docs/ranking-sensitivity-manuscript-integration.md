# Ranking-sensitivity figure in the Metrics Paper

## Recommended role and placement

Use the light-mode global Plotnine figure as a main-text result, immediately after the Introduction
paragraph that explains why heterogeneous evaluation protocols make model comparisons difficult and
before the paragraph beginning “To structure this discussion”. This location gives the figure a
clear argumentative role:

1. benchmark studies reach discordant conclusions;
2. the new analysis shows directly that reasonable, literature-supported evaluation choices change
   the selected model;
3. the manuscript then introduces the taxonomy needed to describe those choices systematically.

The existing supplementary scPerturBench slope figure can remain as external corroboration based on
published benchmark results. It should no longer be the sole figure cited for the general claim that
evaluation design changes rankings.

The manuscript should consume the vector asset
`ranking-sensitivity-global-light.pdf` at `\textwidth`. The reproducibility repository remains the
source of truth for the figure, its five figure-specific source-data CSVs and its analysis manifest.
MetricHub should contain only the selected PDF plus the corresponding LaTeX text and caption.

## Caption draft

**Literature-supported evaluation choices frequently change which perturbation-response model wins.**
The three benchmark tasks are analyzed separately and never pooled. **(a)** Fraction of all unordered
pairs of literature-eligible evaluation protocols that select different winning models. Labels show
the number of disagreeing protocol pairs divided by all eligible protocol pairs. Each pair is
evaluated on the same ground-truth-defined evaluation units and must retain at least 75% common
coverage in every fold. **(b)** Fractional winner share across eligible protocols under the standard
mean reduction across perturbations. Every protocol receives equal weight and tied winners divide one
vote equally; simple baselines are shown in gray. **(c)** Winner-change rate for matched protocols that
differ in one evaluation factor while holding the remaining protocol dimensions fixed. Cells show the
number of winner changes divided by the number of matched contrasts and are displayed only when at
least 10 contrasts are available. An evaluation-family contrast may require a compatible change in
the score function; all other rows represent strict one-factor contrasts. **(d)** The 12 one-factor
contrasts observed across the largest numbers of matched protocol backgrounds. Points show the
winner-change rate and labels give winner changes divided by matched backgrounds. Protocol families
were eligible when their semantic evaluation family occurred in at least two independent scholarly
works, after collapsing ranking-equivalent implementations. Abbreviations: DE, differential
expression; HVG, highly variable gene.

## Proposed manuscript edits after figure approval

- Replace the Introduction’s parenthetical reliance on the supplementary scPerturBench figure with a
  direct statement of the three task-level winner-disagreement rates and cite the new figure.
- Add the new figure environment after that paragraph with label
  `fig:evaluation-ranking-sensitivity`.
- Keep the taxonomy figure immediately after the following paragraph; LaTeX can renumber it
  automatically.
- Retain the selected bump plots as supplementary illustrative examples. They explain *how* rankings
  change, whereas the main global figure establishes *how often* this occurs.
- Keep the dark-mode figure outside the manuscript and use it only in presentations.
