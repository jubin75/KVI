"""
Slot Registry + Fact Type classifier (runtime, lightweight)

Design goal:
- Slots are NOT the universe of knowledge; they are the minimal set of *adjudicable* facts.
- We classify queries into Fact Types, then check whether any adjudicable slots cover them.
- Coverage failure => explicit downgrade (fail-closed): do NOT inject schema; L0 explains "schema does not cover this fact type";
  L1 is controlled; L2 disabled or heavily constrained.

This module is intentionally heuristic and dependency-free.
It must NOT modify retrieval/injection internals; it only provides routing signals.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Set


@dataclass(frozen=True)
class SlotSpec:
    slot_name: str
    fact_type: str
    evidence_density: str = "medium"  # high|medium|low
    stability: str = "medium"  # high|medium|low
    temporal_sensitivity: str = "low"  # low|medium|high
    adjudicable: bool = True


@dataclass(frozen=True)
class FactTypePolicy:
    """
    What we are allowed to do when schema slot coverage is none.
    """

    fact_type: str
    # Whether L1 can answer this using domain prior without evidence.
    allow_domain_prior: bool = True
    # Whether L2 is allowed at all (default should be conservative).
    allow_speculative: bool = False


# Minimal registry for current system slots (extend offline via Slot Proposal workflow).
SLOT_REGISTRY: Dict[str, SlotSpec] = {
    # Epidemiology / transmission
    "transmission": SlotSpec(
        slot_name="transmission",
        fact_type="epidemiology.transmission",
        evidence_density="high",
        stability="high",
        temporal_sensitivity="low",
        adjudicable=True,
    ),
    # Pathogenesis / mechanism (often low stability; still adjudicable when evidence exists)
    "pathogenesis": SlotSpec(
        slot_name="pathogenesis",
        fact_type="pathogenesis.mechanism",
        evidence_density="medium",
        stability="medium",
        temporal_sensitivity="medium",
        adjudicable=True,
    ),
    "mechanism": SlotSpec(
        slot_name="mechanism",
        fact_type="pathogenesis.mechanism",
        evidence_density="medium",
        stability="medium",
        temporal_sensitivity="medium",
        adjudicable=True,
    ),
    # Diagnostics / prevention / treatment / epi
    "diagnosis": SlotSpec("diagnosis", "clinical.diagnosis", evidence_density="medium", stability="high", temporal_sensitivity="low"),
    "prevention": SlotSpec("prevention", "public_health.prevention", evidence_density="medium", stability="high", temporal_sensitivity="low"),
    "treatment": SlotSpec("treatment", "clinical.treatment", evidence_density="low", stability="medium", temporal_sensitivity="high"),
    "epidemiology": SlotSpec("epidemiology", "epidemiology.general", evidence_density="medium", stability="medium", temporal_sensitivity="medium"),
    "clinical_features": SlotSpec("clinical_features", "clinical.features", evidence_density="medium", stability="high", temporal_sensitivity="low"),
    "risk_factors": SlotSpec("risk_factors", "epidemiology.risk_factors", evidence_density="medium", stability="medium", temporal_sensitivity="medium"),
    "complications": SlotSpec("complications", "clinical.complications", evidence_density="medium", stability="medium", temporal_sensitivity="medium"),
    "prognosis": SlotSpec("prognosis", "clinical.prognosis", evidence_density="medium", stability="medium", temporal_sensitivity="high"),
    "overview": SlotSpec("overview", "definition.overview", evidence_density="low", stability="high", temporal_sensitivity="low"),
    # Proposed slots (adjudicable, stable). These reduce fail-closed frequency for common questions.
    "disease_full_name": SlotSpec(
        slot_name="disease_full_name",
        fact_type="taxonomy.definition",
        evidence_density="high",
        stability="high",
        temporal_sensitivity="low",
        adjudicable=True,
    ),
    "geographic_distribution": SlotSpec(
        slot_name="geographic_distribution",
        fact_type="epidemiology.geography",
        evidence_density="medium",
        stability="medium",
        temporal_sensitivity="high",
        adjudicable=True,
    ),
}


# Fact types that users often ask but are NOT covered by current schema slots.
FACT_TYPE_POLICIES: Dict[str, FactTypePolicy] = {
    # Definition/abbreviation expansion is generally stable & safe as domain-prior.
    "taxonomy.definition": FactTypePolicy("taxonomy.definition", allow_domain_prior=True, allow_speculative=False),
    # Geography distribution is often data-sensitive; without evidence we should be conservative.
    "epidemiology.geography": FactTypePolicy("epidemiology.geography", allow_domain_prior=False, allow_speculative=False),
}


def classify_fact_types(query: str) -> Set[str]:
    """
    Heuristic fact-type classifier from user query text.
    Returns a set of fact_type strings.
    """
    q = (query or "").strip().lower()
    if not q:
        return set()
    out: Set[str] = set()

    # taxonomy.definition: full name / abbreviation expansion
    if re.search(r"(全称|全名|缩写|展开|英文全称|full\s*name|stand\s*for|what\s+does\s+\w+\s+stand\s+for)", q, re.I):
        out.add("taxonomy.definition")

    # epidemiology.geography: where in China / region distribution
    if re.search(r"(哪些地方|哪里|在哪些省|地区分布|分布在|中国.*哪里|where.*china|geograph|province)", q, re.I):
        out.add("epidemiology.geography")

    return out


def adjudicable_slots_for_query(*, inferred_slots: Set[str], fact_types: Set[str]) -> Set[str]:
    """
    Return the subset of inferred_slots that are adjudicable and relevant.
    Current rule: a slot is adjudicable if it's in registry and adjudicable==True.
    Fact types can be used for future routing (e.g., require slot fact_type in fact_types),
    but for now we keep it minimal and conservative.
    """
    out: Set[str] = set()
    for s in inferred_slots:
        spec = SLOT_REGISTRY.get(str(s))
        if spec and bool(spec.adjudicable):
            out.add(str(s))
    return out


def fact_types_need_schema_coverage(fact_types: Set[str]) -> Set[str]:
    """
    Fact types that are expected to be adjudicated by schema slots.
    If these are asked but no adjudicable slots are inferred, we should fail-closed.
    """
    # For now, any recognized fact type triggers the coverage check.
    return set(fact_types or set())


def domain_prior_allowed_for_fact_types(fact_types: Set[str]) -> bool:
    """
    If any asked fact_type forbids domain-prior without evidence, return False.
    """
    for ft in (fact_types or set()):
        pol = FACT_TYPE_POLICIES.get(ft)
        if pol is not None and not bool(pol.allow_domain_prior):
            return False
    return True


def speculative_allowed_for_fact_types(fact_types: Set[str]) -> bool:
    for ft in (fact_types or set()):
        pol = FACT_TYPE_POLICIES.get(ft)
        if pol is not None and not bool(pol.allow_speculative):
            return False
    return True


