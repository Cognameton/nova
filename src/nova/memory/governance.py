"""Governance helpers for higher-order memory records."""

from __future__ import annotations

from typing import Any

from nova.types import MemoryEvent


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "our",
    "remains",
    "stay",
    "the",
    "this",
    "to",
    "toward",
    "we",
}
NEGATION_CUES = (" no ", " not ", " never ", " don't ", " do not ", " no longer ", " used to ")
PREFIXES = (
    "user preferences:",
    "user values:",
    "nova identity:",
    "nova values:",
    "relationship context:",
    "continuity focus:",
    "identity milestone:",
    "identity continuity:",
    "identity shift:",
    "identity tension:",
    "relationship milestone:",
    "relationship continuity:",
    "relationship shift:",
    "relationship tension:",
    "value milestone:",
    "value continuity:",
    "value shift:",
    "value tension:",
)


def normalize_governed_event(event: MemoryEvent) -> MemoryEvent:
    """Attach default governance/provenance metadata to higher-order events."""

    metadata = dict(event.metadata or {})
    source_ids = [value for value in metadata.get("source_event_ids", []) or [] if value]
    source_channels = [value for value in metadata.get("source_channels", []) or [] if value]
    evidence_preview = [value for value in metadata.get("evidence_preview", []) or [] if value]
    governance_scope = str(
        metadata.get("governance_scope")
        or metadata.get("theme")
        or metadata.get("self_state_version")
        or event.kind
        or event.channel
    )
    claim_axis = str(metadata.get("claim_axis") or governance_scope)
    claim_value = str(metadata.get("claim_value") or _claim_value(event.text))
    support_count = max(
        int(metadata.get("support_count", 0) or 0),
        int(metadata.get("event_count", 0) or 0),
        int(metadata.get("distinct_turn_count", 0) or 0),
        len(source_ids),
    )

    metadata["governance_scope"] = governance_scope
    metadata["claim_axis"] = claim_axis
    metadata["claim_value"] = claim_value
    metadata["support_count"] = support_count
    metadata["provenance"] = {
        "source_event_ids": sorted(set(source_ids)),
        "source_channels": sorted(set(source_channels)),
        "evidence_preview": evidence_preview[:3],
        "support_count": support_count,
    }
    metadata.setdefault("governance_status", "active")

    return MemoryEvent(
        schema_version=event.schema_version,
        event_id=event.event_id,
        timestamp=event.timestamp,
        session_id=event.session_id,
        turn_id=event.turn_id,
        channel=event.channel,
        kind=event.kind,
        text=event.text,
        summary=event.summary,
        tags=list(event.tags),
        importance=event.importance,
        confidence=event.confidence,
        continuity_weight=event.continuity_weight,
        retention=event.retention,
        supersedes=list(event.supersedes),
        source=event.source,
        metadata=metadata,
    )


def payload_governance_scope(payload: dict[str, Any]) -> str:
    metadata = dict(payload.get("metadata", {}) or {})
    return str(metadata.get("governance_scope") or metadata.get("theme") or payload.get("kind") or "")


def payload_support_count(payload: dict[str, Any]) -> int:
    metadata = dict(payload.get("metadata", {}) or {})
    provenance = dict(metadata.get("provenance", {}) or {})
    return max(
        int(metadata.get("support_count", 0) or 0),
        int(metadata.get("event_count", 0) or 0),
        int(metadata.get("distinct_turn_count", 0) or 0),
        int(provenance.get("support_count", 0) or 0),
        len(list(metadata.get("source_event_ids", []) or [])),
        len(list(provenance.get("source_event_ids", []) or [])),
    )


def payload_conflicts(existing: dict[str, Any], incoming: MemoryEvent) -> bool:
    existing_metadata = dict(existing.get("metadata", {}) or {})
    incoming_metadata = dict(incoming.metadata or {})
    existing_axis = str(existing_metadata.get("claim_axis") or payload_governance_scope(existing))
    incoming_axis = str(incoming_metadata.get("claim_axis") or payload_governance_scope(incoming.to_dict()))
    if not existing_axis or existing_axis != incoming_axis:
        return False

    existing_value = str(existing_metadata.get("claim_value") or _claim_value(str(existing.get("text") or "")))
    incoming_value = str(incoming_metadata.get("claim_value") or _claim_value(incoming.text))
    if not existing_value or not incoming_value:
        return False
    if existing_value == incoming_value:
        return False

    explicit_existing = str(existing_metadata.get("claim_value") or "")
    explicit_incoming = str(incoming_metadata.get("claim_value") or "")
    if explicit_existing and explicit_incoming:
        return True

    existing_tokens = set(existing_value.split())
    incoming_tokens = set(incoming_value.split())
    overlap = len(existing_tokens.intersection(incoming_tokens))
    if overlap >= 2 and _polarity(str(existing.get("text") or "")) != _polarity(incoming.text):
        return True
    return False


def merge_provenance(existing_metadata: dict[str, Any], incoming_metadata: dict[str, Any]) -> dict[str, Any]:
    existing_provenance = dict(existing_metadata.get("provenance", {}) or {})
    incoming_provenance = dict(incoming_metadata.get("provenance", {}) or {})
    source_event_ids = sorted(
        {
            *list(existing_metadata.get("source_event_ids", []) or []),
            *list(incoming_metadata.get("source_event_ids", []) or []),
            *list(existing_provenance.get("source_event_ids", []) or []),
            *list(incoming_provenance.get("source_event_ids", []) or []),
        }
    )
    source_channels = sorted(
        {
            *list(existing_metadata.get("source_channels", []) or []),
            *list(incoming_metadata.get("source_channels", []) or []),
            *list(existing_provenance.get("source_channels", []) or []),
            *list(incoming_provenance.get("source_channels", []) or []),
        }
    )
    evidence_preview = list(existing_provenance.get("evidence_preview", []) or [])
    for preview in list(incoming_provenance.get("evidence_preview", []) or []):
        if preview and preview not in evidence_preview:
            evidence_preview.append(preview)
    support_count = max(
        int(existing_metadata.get("support_count", 0) or 0),
        int(incoming_metadata.get("support_count", 0) or 0),
        int(existing_provenance.get("support_count", 0) or 0),
        int(incoming_provenance.get("support_count", 0) or 0),
        len(source_event_ids),
    )
    return {
        "source_event_ids": source_event_ids,
        "source_channels": source_channels,
        "evidence_preview": evidence_preview[:3],
        "support_count": support_count,
    }


def _claim_value(text: str) -> str:
    normalized = f" {' '.join(text.lower().split())} "
    for prefix in PREFIXES:
        if normalized.strip().startswith(prefix):
            normalized = f" {normalized.strip()[len(prefix):].strip()} "
            break
    normalized = normalized.replace(";", " ").replace(",", " ").replace(".", " ")
    tokens = [
        token
        for token in normalized.split()
        if token and token not in STOPWORDS and token not in {"i", "my"}
    ]
    return " ".join(tokens[:12])


def _polarity(text: str) -> str:
    normalized = f" {' '.join(text.lower().split())} "
    return "negative" if any(cue in normalized for cue in NEGATION_CUES) else "positive"
