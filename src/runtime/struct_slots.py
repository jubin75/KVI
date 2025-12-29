from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import re


@dataclass
class EvidenceSchema:
    """
    Minimal, stable intermediate representation (slots) between evidence and generation.

    This is deliberately NOT a natural-language answer. It is a constraint summary derived
    from selected evidence blocks, used for injection/guidance.
    """

    transmission_primary: List[str] = field(default_factory=list)
    transmission_secondary: List[str] = field(default_factory=list)
    vector: List[str] = field(default_factory=list)
    pathogenesis_notes: List[str] = field(default_factory=list)

    def dedup(self) -> "EvidenceSchema":
        def _dedup(xs: List[str]) -> List[str]:
            out: List[str] = []
            seen = set()
            for x in xs:
                k = re.sub(r"\s+", " ", x.strip().lower())
                if not k or k in seen:
                    continue
                seen.add(k)
                out.append(x.strip())
            return out

        self.transmission_primary = _dedup(self.transmission_primary)
        self.transmission_secondary = _dedup(self.transmission_secondary)
        self.vector = _dedup(self.vector)
        self.pathogenesis_notes = _dedup(self.pathogenesis_notes)
        return self


def build_schema_from_evidence_texts(texts: Iterable[str]) -> EvidenceSchema:
    """
    Extract + normalize + deduplicate. Do NOT generate new facts.
    Heuristic, domain-biased to infectious diseases (SFTSV/SARS2-like).
    """
    schema = EvidenceSchema()

    for t in texts:
        if not isinstance(t, str) or not t.strip():
            continue
        tl = t.lower()

        # --- Transmission / vectors ---
        if ("tick" in tl) or ("tick-borne" in tl) or ("蜱" in t) or ("叮咬" in t):
            schema.transmission_primary.append("tick bite (蜱叮咬)")
        if ("haemaphysalis" in tl) or ("h. longicornis" in tl) or ("长角血蜱" in t) or ("长角蜱" in t):
            schema.vector.append("Haemaphysalis (e.g., H. longicornis)")
        if ("human-to-human" in tl) or ("person-to-person" in tl) or ("人传人" in t):
            schema.transmission_secondary.append("human-to-human transmission (人传人)")
        if ("body fluid" in tl) or ("blood" in tl) or ("体液" in t) or ("血液" in t):
            schema.transmission_secondary.append("contact with blood/body fluids (血液/体液接触)")
        if ("aerosol" in tl) or ("droplet" in tl) or ("airborne" in tl) or ("气溶胶" in t) or ("飞沫" in t):
            schema.transmission_secondary.append("respiratory droplets/aerosol (飞沫/气溶胶)")

        # --- Pathogenesis / mechanism notes (keep as short evidence-derived hints) ---
        # Keep these as "notes" rather than asserting causality, unless the evidence explicitly states it.
        if ("cytokine storm" in tl) or ("cytokine" in tl) or ("细胞因子风暴" in t) or ("细胞因子" in t):
            schema.pathogenesis_notes.append("cytokines / cytokine storm (细胞因子/风暴)")
        if ("immune dysfunction" in tl) or ("immune" in tl) or ("immun" in tl) or ("免疫" in t):
            schema.pathogenesis_notes.append("immune response/dysregulation (免疫反应/失衡)")
        if ("interferon" in tl) or re.search(r"\bifn\b", tl) or ("干扰素" in t):
            schema.pathogenesis_notes.append("interferon/IFN signaling (干扰素通路)")
        if ("inflammation" in tl) or ("inflammatory" in tl) or ("炎症" in t) or ("过度炎症" in t):
            schema.pathogenesis_notes.append("inflammation / hyperinflammation (炎症/过度炎症)")
        if ("mods" in tl) or ("multi-organ" in tl) or ("多器官" in t) or ("器官功能衰竭" in t):
            schema.pathogenesis_notes.append("multi-organ dysfunction (MODS/多器官功能障碍)")

    return schema.dedup()


def schema_to_injection_text(schema: EvidenceSchema, *, max_items_per_field: int = 3) -> str:
    """
    Produce a compact, abstracted constraint summary to inject/generate from.
    This text is the "schema interface", not evidence quotes.
    """
    def _fmt(xs: List[str]) -> str:
        xs = [x.strip() for x in xs if isinstance(x, str) and x.strip()]
        if not xs:
            return "N/A"
        xs = xs[: max(1, int(max_items_per_field))]
        return "; ".join(xs)

    return (
        "Confirmed evidence slots (do NOT treat as a full answer; do NOT invent new facts):\n"
        f"- transmission_primary: {_fmt(schema.transmission_primary)}\n"
        f"- transmission_secondary: {_fmt(schema.transmission_secondary)}\n"
        f"- vector: {_fmt(schema.vector)}\n"
        f"- pathogenesis_notes: {_fmt(schema.pathogenesis_notes)}\n"
    )


