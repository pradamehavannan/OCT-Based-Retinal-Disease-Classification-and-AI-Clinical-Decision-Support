# Knowledge base — curated reference passages for Part 2 (RAG narrator)

This directory holds the **only** material the Part 2 narrator is allowed to
ground its explanations in. There is no live internet retrieval.

Each `*.md` file (except `_TEMPLATE.md` and files starting with `_`) is one
**entry**. `src/oct_cds/rag/ingest.py` splits every entry into passages — one per
`##` heading — and those passages are what the LLM sees, quotes, and cites.

## Licensing — hard rules

Every entry must be built **only** from:

- **NEI (U.S. National Eye Institute)** material — public domain (U.S. federal work), or
- **open-access articles under a licence that permits derivative works** — CC-BY,
  CC-BY-SA. **CC-BY-NC-ND does not qualify** (no derivatives) — so **StatPearls is
  out**.

Explicitly **not allowed** as sources: AAO Preferred Practice Patterns, EyeWiki,
or any content whose terms restrict use "in an artificial intelligence program".

All text must be **paraphrased / summarised in your own words**, never copied
verbatim. Every entry lists its real sources in front-matter and in
[`SOURCES.md`](SOURCES.md).

## Entry format

Front matter (YAML) + Markdown body. See [`_TEMPLATE.md`](_TEMPLATE.md).

```yaml
---
id: amd                      # unique slug; becomes the passage-id prefix
title: Age-related Macular Degeneration
covers: [AMD, Drusen, CNV]   # Part 1 class keys this entry is THE reference for
kb_version: 1
sources:
  - name: "National Eye Institute — Age-Related Macular Degeneration"
    url: "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/age-related-macular-degeneration"
    licence: "public-domain (U.S. federal)"
  - name: "Author et al. (2023). Title. Journal. doi:10.xxxx/xxxxx"
    url: "https://doi.org/10.xxxx/xxxxx"
    licence: "CC-BY-4.0"
---
```

Body — one thought per `##` section, self-contained (a clinician should get a
complete point from a single passage). Recommended sections, in this order:

```markdown
## Overview
## OCT features
## Clinical significance
## Management and referral
```

Sections are flexible, but keep them focused and titled — the title is shown to
the clinician as the citation label.

## The `covers` mapping

`covers` maps an entry to the Part 1 disease classes it is the reference for.
Retrieval is **decision-aware**: given `Recommendation.predicted_class`, the
narrator always pulls that class's entry, then adds semantic hits for the
differential classes.

Requirements (validated at ingest — the build fails otherwise):

- The 7 **pathology** classes — `AMD`, `CNV`, `CSR`, `DME`, `DR`, `Drusen`,
  `Macular Hole` — must each appear in **exactly one** entry's `covers`. One
  entry may cover several classes (e.g. an AMD entry covering `AMD`, `Drusen`,
  `CNV`).
- An entry may have an **empty `covers`** — a general reference (e.g. "reading a
  macular OCT"). It is retrieved semantically only, never as a class's designated
  entry.
- `Normal` must **not** appear in any `covers`. A `Normal` prediction skips RAG
  entirely and passes through Part 1's plain "no pathology detected" output — no
  LLM call.

## Passage ids & citations

- Passage id: `{entry id}#{heading slug}` — e.g. `amd#oct-features`.
- The narrator cites inline as `[amd#oct-features]`; the report expands each cited
  passage to its entry's `sources` in a References block.
- Ids are deterministic (file + heading), so citations and the narrative cache
  stay stable across runs.

## Index

The FAISS index is a build artifact (`knowledge_base/.index/`, git-ignored),
rebuilt from the `*.md` files whenever their content or `kb_version` changes.
Nothing here needs a GPU; embedding ~30 short passages is a few seconds on CPU.
