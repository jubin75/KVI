from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


EvidenceStatus = Literal["draft", "reviewed", "approved", "rejected"]

# NOTE: semantic_type is intentionally small/frozen to prevent routing-rule explosion.
SemanticType = Literal[
    "drug",
    "symptom",
    "location",
    "procedure",
    "laboratory",
    "population",
    "outcome",
    "generic",
]

Polarity = Literal["positive", "negative", "neutral"]


class EvidenceRejectionCode:
    """
    Frozen finite set (docs/11_Knowledge_Authoring_Layer.md).
    Do NOT introduce per-schema custom codes.
    """

    SEMANTIC_TYPE_MISMATCH = "SEMANTIC_TYPE_MISMATCH"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    NON_ENUMERATIVE = "NON_ENUMERATIVE"
    MIXED_SEMANTICS = "MIXED_SEMANTICS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NOT_APPROVED = "NOT_APPROVED"


@dataclass(frozen=True)
class EvidenceRejection:
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": str(self.code),
            "message": str(self.message),
            "details": dict(self.details or {}),
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class Provenance:
    source_type: str = "guideline"
    organization: Optional[str] = None
    document_title: str = ""
    publication_year: Optional[int] = None
    page_range: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": str(self.source_type or ""),
            "organization": self.organization,
            "document_title": str(self.document_title or ""),
            "publication_year": int(self.publication_year) if self.publication_year is not None else None,
            "page_range": self.page_range,
        }


@dataclass(frozen=True)
class ExternalRefs:
    document_id: Optional[str] = None
    pmid: Optional[str] = None
    orcid: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    published_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "pmid": self.pmid,
            "orcid": self.orcid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": list(self.authors or []),
            "published_at": self.published_at,
        }


@dataclass
class EvidenceUnit:
    """
    Authoring-layer EvidenceUnit (human writable/auditable).

    Notes:
    - `claim` is the canonical text field (docs/11).
      For back-compat with older drafts you may also provide `semantic_text`,
      which will be normalized into `claim` at parse time.
    """

    evidence_id: str
    semantic_type: SemanticType
    schema_id: str
    claim: str
    polarity: Polarity = "neutral"
    slot_projection: Dict[str, List[str]] = field(default_factory=dict)
    status: EvidenceStatus = "draft"
    provenance: Provenance = field(default_factory=Provenance)
    external_refs: ExternalRefs = field(default_factory=ExternalRefs)

    # Extra fields seen in docs/012 (kept here to ease migration; not required at runtime)
    evidence_type: str = "clinical_guideline"
    review_feedback: Optional[Dict[str, Any]] = None
    # Frozen rejection model (backend -> frontend). Only meaningful when status == "rejected".
    # Kept as dict for forward-compat with UI; structure must follow EvidenceRejection.to_dict().
    rejection: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def normalized_text(self) -> str:
        return str(self.claim or "").strip()

    def is_approved(self) -> bool:
        return str(self.status) == "approved"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": str(self.evidence_id),
            "semantic_type": str(self.semantic_type),
            "schema_id": str(self.schema_id),
            "claim": str(self.claim),
            "polarity": str(self.polarity),
            "slot_projection": {str(k): [str(x) for x in (v or []) if str(x).strip()] for k, v in (self.slot_projection or {}).items()},
            "status": str(self.status),
            "provenance": self.provenance.to_dict() if isinstance(self.provenance, Provenance) else (self.provenance or {}),
            "external_refs": self.external_refs.to_dict() if isinstance(self.external_refs, ExternalRefs) else (self.external_refs or {}),
            # migration-friendly extras
            "evidence_type": str(self.evidence_type or ""),
            "review_feedback": self.review_feedback,
            "rejection": self.rejection,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "EvidenceUnit":
        if not isinstance(obj, dict):
            raise TypeError("EvidenceUnit.from_dict expects a dict")

        # Accept either `claim` (preferred) or `semantic_text` (legacy)
        claim = obj.get("claim", None)
        if not isinstance(claim, str) or not claim.strip():
            st = obj.get("semantic_text", None)
            claim = str(st or "")

        prov = obj.get("provenance") if isinstance(obj.get("provenance"), dict) else {}
        exr = obj.get("external_refs") if isinstance(obj.get("external_refs"), dict) else {}

        return cls(
            evidence_id=str(obj.get("evidence_id") or obj.get("id") or ""),
            semantic_type=str(obj.get("semantic_type") or "generic").lower(),  # type: ignore[arg-type]
            schema_id=str(obj.get("schema_id") or obj.get("schema") or ""),
            claim=str(claim or ""),
            polarity=str(obj.get("polarity") or "neutral").lower(),  # type: ignore[arg-type]
            slot_projection=obj.get("slot_projection") if isinstance(obj.get("slot_projection"), dict) else {},
            status=str(obj.get("status") or "draft").lower(),  # type: ignore[arg-type]
            provenance=Provenance(
                source_type=str(prov.get("source_type") or "guideline"),
                organization=prov.get("organization", None),
                document_title=str(prov.get("document_title") or ""),
                publication_year=(int(prov["publication_year"]) if prov.get("publication_year") is not None else None),
                page_range=prov.get("page_range", None),
            ),
            external_refs=ExternalRefs(
                document_id=exr.get("document_id", None),
                pmid=exr.get("pmid", None),
                orcid=exr.get("orcid", None),
                title=exr.get("title", None),
                abstract=exr.get("abstract", None),
                authors=[str(a) for a in (exr.get("authors") or []) if str(a).strip()] if isinstance(exr.get("authors"), list) else [],
                published_at=exr.get("published_at", None),
            ),
            evidence_type=str(obj.get("evidence_type") or obj.get("type") or "clinical_guideline"),
            review_feedback=obj.get("review_feedback") if isinstance(obj.get("review_feedback"), dict) else None,
            rejection=obj.get("rejection") if isinstance(obj.get("rejection"), dict) else None,
            created_at=(str(obj["created_at"]) if obj.get("created_at") is not None else None),
            updated_at=(str(obj["updated_at"]) if obj.get("updated_at") is not None else None),
        )


# Fields that are allowed to be shipped to runtime KVBank.
# Authoring-only fields like status/review_feedback are intentionally excluded.
APPROVED_ONLY_FIELDS = (
    "evidence_id",
    "semantic_type",
    "schema_id",
    "claim",
    "polarity",
    "slot_projection",
    "provenance",
    "external_refs",
    "evidence_type",
)


@dataclass(frozen=True)
class RuntimeEvidenceRecord:
    """
    Runtime-safe Evidence record (approved-only projection).
    This is what should be written into the FAISS KVBank metadata layer.
    """

    evidence_id: str
    semantic_text: str
    evidence_type: str
    semantic_type: str
    schema_id: str
    polarity: str
    slot_projection: Dict[str, List[str]] = field(default_factory=dict)
    external_refs: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    contract: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": str(self.evidence_id),
            "semantic_text": str(self.semantic_text or ""),
            "evidence_type": str(self.evidence_type or ""),
            "semantic_type": str(self.semantic_type or ""),
            "schema_id": str(self.schema_id or ""),
            "polarity": str(self.polarity or "neutral"),
            "slot_projection": {str(k): [str(x) for x in (v or []) if str(x).strip()] for k, v in (self.slot_projection or {}).items()},
            "external_refs": dict(self.external_refs or {}),
            "provenance": dict(self.provenance or {}),
            "contract": dict(self.contract or {}),
        }


def new_evidence_id(*, prefix: str = "EVIDENCE") -> str:
    """
    Generate a stable, human-friendly evidence id (UUID-backed).
    """
    u = uuid.uuid4().hex
    return f"{prefix}_{u}"


def to_runtime_record(
    eu: EvidenceUnit,
    *,
    allowed_injection: bool = True,
    list_only_allowed: bool = True,
) -> RuntimeEvidenceRecord:
    """
    Convert an Authoring EvidenceUnit into a runtime-safe record.
    """
    return RuntimeEvidenceRecord(
        evidence_id=str(eu.evidence_id),
        semantic_text=str(eu.normalized_text()),
        evidence_type=str(eu.evidence_type or "clinical_guideline"),
        semantic_type=str(eu.semantic_type or "generic"),
        schema_id=str(eu.schema_id or ""),
        polarity=str(eu.polarity or "neutral"),
        slot_projection={str(k): [str(x) for x in (v or []) if str(x).strip()] for k, v in (eu.slot_projection or {}).items()},
        external_refs=eu.external_refs.to_dict() if isinstance(eu.external_refs, ExternalRefs) else (eu.external_refs or {}),
        provenance=eu.provenance.to_dict() if isinstance(eu.provenance, Provenance) else (eu.provenance or {}),
        contract={"allowed_injection": bool(allowed_injection), "list_only_allowed": bool(list_only_allowed)},
    )


def validate_evidence_for_injection(
    *,
    evidence: RuntimeEvidenceRecord,
    target_slot_semantic_type: Optional[str],
    active_schema_id: Optional[str],
    min_confidence: float = 0.0,
    similarity_confidence: Optional[float] = None,
) -> Tuple[bool, Optional[EvidenceRejection]]:
    """
    Hard contract checks before injection/slot-filling (docs/11):
    - semantic_type matches slot.semantic_type
    - schema_id matches active_schema
    - (optional) similarity confidence above threshold
    """
    if similarity_confidence is not None and float(similarity_confidence) < float(min_confidence):
        return (
            False,
            EvidenceRejection(
                code=EvidenceRejectionCode.LOW_CONFIDENCE,
                message="Evidence similarity/confidence is below threshold.",
                details={"min_confidence": float(min_confidence), "actual": float(similarity_confidence)},
                confidence=0.8,
            ),
        )

    if active_schema_id and str(evidence.schema_id or "").strip() and str(evidence.schema_id) != str(active_schema_id):
        return (
            False,
            EvidenceRejection(
                code=EvidenceRejectionCode.SCHEMA_MISMATCH,
                message="Evidence schema_id does not match active schema.",
                details={"expected": str(active_schema_id), "actual": str(evidence.schema_id)},
                confidence=0.95,
            ),
        )

    if target_slot_semantic_type and str(target_slot_semantic_type).strip():
        exp = str(target_slot_semantic_type).strip().lower()
        act = str(evidence.semantic_type or "").strip().lower()
        if act and exp and act != exp:
            return (
                False,
                EvidenceRejection(
                    code=EvidenceRejectionCode.SEMANTIC_TYPE_MISMATCH,
                    message="Evidence semantic_type does not match target slot.",
                    details={"expected": exp, "actual": act},
                    confidence=0.93,
                ),
            )

    # contract flag
    if isinstance(evidence.contract, dict) and evidence.contract.get("allowed_injection") is False:
        return (
            False,
            EvidenceRejection(
                code=EvidenceRejectionCode.NOT_APPROVED,
                message="Evidence is not allowed for injection by contract.",
                details={},
                confidence=0.9,
            ),
        )

    return True, None

