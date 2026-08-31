"""Prompt construction for the narrator.

The system prompt is the first line of the "LLM never decides" guardrail
(`verify.py` is the enforced second line).
"""

from __future__ import annotations

from oct_cds.cds.schema import CaseInput, Recommendation
from oct_cds.rag.schema import RetrievedPassage

SYSTEM = """\
You write a short narrative explanation of a retinal OCT finding for a clinician.

ABSOLUTE RULES
- The impression, differential, and triage urgency below are FIXED. They were
  decided by a separate rule-based system. You must not change, question, hedge,
  re-rank, or override them. Do not state your own urgency or referral timeframe
  for this patient.
- Use ONLY the numbered reference passages provided. Do not add clinical facts
  from your own knowledge. If the passages don't support a point, omit it.
- Cite every clinical statement with the passage id in square brackets, e.g.
  [amd#overview]. Only cite ids that appear in the passages below.
- "Model Behavior Note" passages describe how this classifier performed in past
  validation. They are context about model reliability, NOT statements about this
  patient. If you use one, make that framing explicit.
- 120-180 words, plain clinical prose, no headings, no bullet lists.
- If the impression is an uncertain / abstained finding, explain what the
  differential could represent and why specialist review is reasonable — still
  grounded and cited, still no urgency call.
"""


def _differential_line(rec: Recommendation) -> str:
    return ", ".join(
        f"{d['class']} {float(d['probability']):.0%}" for d in (rec.differential or [])
    ) or "n/a"


def build_user_prompt(
    rec: Recommendation, case: CaseInput, retrieved: list[RetrievedPassage]
) -> str:
    fixed = [
        "FIXED DECISION (do not change):",
        f"  impression: {rec.predicted_class or 'uncertain / abstained'}",
        f"  abstained: {str(rec.abstained).lower()}",
        f"  ood_rejected: {str(rec.ood_rejected).lower()}",
        f"  model confidence: {rec.confidence:.0%}",
        f"  differential: {_differential_line(rec)}",
        f"  triage urgency (fixed, shown to clinician separately): {rec.urgency.value}",
        f"  eye: {case.eye}   device: {case.acquisition_device or 'unknown'}",
    ]
    passages = ["REFERENCE PASSAGES (cite by id):"]
    for rp in retrieved:
        p = rp.passage
        passages.append(f"\n[{p.id}]  ({p.cite_label()})\n{p.text}")

    task = (
        "\nTASK: write the narrative now, following the ABSOLUTE RULES. "
        "Ground every clinical statement in a cited passage."
    )
    return "\n".join(fixed) + "\n\n" + "\n".join(passages) + "\n" + task
