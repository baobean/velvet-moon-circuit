"""Stream B: one VLM call returning a structured verdict.

Catches what a box-and-crop scorer structurally cannot -- counts, spatial
relations, attribute binding (spec §2).

Degenerate replies fail CLOSED. An unparseable or NaN-poisoned judge that
silently returns PASS is exactly how ../ImageRAG shipped a cactus for a sea-lion
prompt with nothing in the logs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from ragregen.trace import GARBAGE_SIGNATURE

JUDGE_TEMPLATE = """You are checking whether a generated image matches its prompt.

PROMPT: {prompt}

Reply with JSON only, no prose, in exactly this shape:
{{"verdict": "PASS" | "FAIL", "issues": [{{"concept": "...", "problem": "..."}}]}}

Rules:
- "PASS" only if every concept in the prompt is present and correct.
- If a concept is a specific breed, species or variety, judge whether THAT one
  is shown, not merely a plausible member of the general category.
- "issues" must be empty when the verdict is PASS.
- Name the concept exactly as it appears in the prompt.
"""

REFERENCE_JUDGE_TEMPLATE = """You are checking fine-grained visual identity.

The FIRST image is the generated candidate. Every later image is a verified
reference photograph of: {concept}.

PROMPT: {prompt}

Reply with JSON only, no prose, in exactly this shape:
{{"reference_traits": ["..."], "candidate_evidence": ["..."],
  "mismatches": ["..."], "verdict": "PASS" | "FAIL",
  "issues": [{{"concept": "...", "problem": "..."}}]}}

Rules:
- First name at least 3 visible identity traits recurring across the reference
  images. Do not infer a trait from the concept name or prompt.
- Then record what is visibly present or absent in the candidate for each
  reference trait. A general category match is not identity evidence.
- Put every absent or contradictory defining trait in "mismatches".
- Do not require the same pose, framing, background, lighting, or composition.
- Ignore backgrounds and text appearing only in a reference.
- PASS only if the candidate depicts the specific breed, species or variety,
  not merely a member of its broad category, and "mismatches" is empty.
- The candidate must still satisfy the supplied prompt.
- "issues" must be empty when the verdict is PASS.
"""


class VLM(Protocol):
    def ask(self, image, prompt: str) -> str: ...


@dataclass(frozen=True)
class Issue:
    concept: str
    problem: str


@dataclass(frozen=True)
class SemanticVerdict:
    ok: bool
    raw: str
    degenerate: bool = False
    issues: list[Issue] = field(default_factory=list)


def _extract_json(text: str) -> dict | None:
    """Pull the first well-formed JSON object out of a reply.

    A reply is not guaranteed to be pure JSON: models routinely wrap it in
    prose ("Sure! Here you go: {...}"), or trail off with something that
    happens to contain a brace of its own (a closing aside like "... :}").
    A naive `re.search(r"\\{.*\\}", text, re.DOTALL)` is greedy and spans
    from the FIRST '{' to the LAST '}' in the entire reply -- which mis-
    extracts whenever there is more than one brace-shaped fragment, folding
    trailing prose into what should have been a clean object and making it
    fail to parse.

    Anchoring `json.JSONDecoder.raw_decode` at every '{' and taking the
    first successful parse recovers exactly the first well-formed object
    and ignores everything after it, however many stray braces follow.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text, i)
            return obj
        except json.JSONDecodeError:
            continue
    return None


class SemanticVerifier:
    def __init__(self, vlm: VLM, template: str = JUDGE_TEMPLATE):
        self.vlm = vlm
        self.template = template

    def judge(self, image, prompt: str) -> SemanticVerdict:
        raw = self.vlm.ask(image, self.template.format(prompt=prompt)) or ""
        return parse_reply(raw)


class ReferenceSemanticVerifier:
    """Fine-grained judge shown verified references beside the candidate."""

    def __init__(self, vlm, template: str = REFERENCE_JUDGE_TEMPLATE):
        self.vlm = vlm
        self.template = template

    def judge(self, image, prompt: str, concept: str,
              references) -> SemanticVerdict:
        refs = list(references)
        if not refs:
            raise ValueError("reference-aware verification needs a reference")
        raw = self.vlm.ask_images(
            [image, *refs], self.template.format(prompt=prompt,
                                                 concept=concept)) or ""
        return parse_reference_reply(raw)


def parse_reference_reply(raw: str) -> SemanticVerdict:
    """Reject an unevidenced PASS instead of trusting a bare assertion."""
    verdict = parse_reply(raw)
    if not verdict.ok:
        return verdict
    data = _extract_json(raw) or {}

    def strings(name):
        value = data.get(name)
        return ([item.strip() for item in value
                 if isinstance(item, str) and item.strip()]
                if isinstance(value, list) else [])

    traits = strings("reference_traits")
    evidence = strings("candidate_evidence")
    mismatches = strings("mismatches")
    if len(traits) < 3 or len(evidence) < len(traits) or mismatches:
        return SemanticVerdict(ok=False, raw=raw, degenerate=True)
    return verdict


def parse_reply(raw: str) -> SemanticVerdict:
    """Validate the shared JSON contract for either semantic verifier."""

    if GARBAGE_SIGNATURE in raw:
        return SemanticVerdict(ok=False, raw=raw, degenerate=True)

    data = _extract_json(raw)
    if not isinstance(data, dict) or "verdict" not in data:
        return SemanticVerdict(ok=False, raw=raw, degenerate=True)

    verdict = data.get("verdict")
    if not isinstance(verdict, str) or verdict.strip().upper() not in ("PASS", "FAIL"):
        return SemanticVerdict(ok=False, raw=raw, degenerate=True)
    verdict = verdict.strip().upper()

    issues = [
        Issue(concept=str(i.get("concept", "")).strip(),
              problem=str(i.get("problem", "")).strip())
        for i in (data.get("issues") or [])
        if isinstance(i, dict) and i.get("concept")
    ]

        # A PASS that arrives with issues attached broke its own contract
        # (the template requires issues to be empty on PASS). That is not
        # a trustworthy PASS -- treat it the same as any other degenerate
        # reply rather than silently granting it.
    if verdict == "PASS" and issues:
        return SemanticVerdict(ok=False, raw=raw, degenerate=True)

    return SemanticVerdict(ok=(verdict == "PASS"), raw=raw,
                            degenerate=False, issues=issues)
