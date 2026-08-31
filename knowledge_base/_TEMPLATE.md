---
id: <slug>                       # unique, lowercase, [a-z0-9-]; e.g. "amd", "macular-hole"
title: <Human-readable disease name>
covers: [<ClassKey>, ...]        # exact Part 1 class keys, e.g. [AMD, Drusen, CNV]
                                 # valid keys: AMD CNV CSR DME DR Drusen "Macular Hole"
                                 # NOT "Normal". Use [] for a general reference entry.
kb_version: 1                    # bump when the content changes
sources:
  - name: "National Eye Institute — <page title>"
    url: "https://www.nei.nih.gov/..."
    licence: "public-domain (U.S. federal)"
  - name: "<Author> et al. (<year>). <Title>. <Journal>. doi:<doi>"
    url: "https://doi.org/<doi>"
    licence: "CC-BY-4.0"          # must permit derivative works
---

<!--
  Paraphrased / summarised only — never copied verbatim.
  One self-contained thought per `##` section; the heading is the citation label.
  Keep each section ~60–150 words.
-->

## Overview

<What the condition is, who it affects, how it presents. Plain clinical language.>

## OCT features

<What this looks like on a macular OCT B-scan — the findings the classifier is
keying on. This section is the most important for grounding the narrative.>

## Clinical significance

<Why it matters: visual prognosis, urgency drivers, what makes a case more or
less concerning.>

## Management and referral

<Typical follow-up / referral pathway and timeframe, at a general level. Do NOT
write anything that could read as a per-patient instruction — the CDS urgency
comes from Part 1's rule engine, not from this text.>
