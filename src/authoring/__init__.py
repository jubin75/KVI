"""
Authoring Layer (Evidence-first).

This package provides:
- A stable EvidenceUnit model for human authoring/review.
- Validation helpers and a narrow rejection-code set.
- Export utilities to produce runtime-safe, approved-only records for KVBank building.

Design constraints (docs frozen):
- Evidence is a semantic entry (no intent/topic routing).
- Only approved evidence may be exported to runtime KVBank.
- Runtime should NOT see review status/feedback; "retrievable == approved".
"""

from .models import (
    APPROVED_ONLY_FIELDS,
    EvidenceRejection,
    EvidenceRejectionCode,
    EvidenceStatus,
    EvidenceUnit,
    RuntimeEvidenceRecord,
    validate_evidence_for_injection,
)

__all__ = [
    "APPROVED_ONLY_FIELDS",
    "EvidenceRejection",
    "EvidenceRejectionCode",
    "EvidenceStatus",
    "EvidenceUnit",
    "RuntimeEvidenceRecord",
    "validate_evidence_for_injection",
]

