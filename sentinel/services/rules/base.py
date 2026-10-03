from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from packages.schemas.common import Severity
from packages.schemas.event import ConditionEvidence
from packages.schemas.rule import RuleDefinition, RuleType
from services.rules.context import EvaluationContext


@dataclass
class RuleMatch:
    rule_key: str
    rule_name: str
    event_type: str
    severity: Severity
    camera_id: str
    ts: datetime
    confidence: float
    track_ids: list[int]
    zone_id: str | None
    zone_name: str | None
    conditions: list[ConditionEvidence] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Rule(Protocol):
    definition: RuleDefinition

    def evaluate(self, ctx: EvaluationContext) -> RuleMatch | None: ...


def build_rule(definition: RuleDefinition) -> Rule:
    from services.rules.builtin import (
        LineCrossRule,
        RestrictedZoneIntrusionRule,
        ZoneDwellRule,
        ZoneEnterRule,
    )

    builders = {
        RuleType.ZONE_ENTER: ZoneEnterRule,
        RuleType.ZONE_DWELL: ZoneDwellRule,
        RuleType.LINE_CROSS: LineCrossRule,
        RuleType.RESTRICTED_ZONE_INTRUSION: RestrictedZoneIntrusionRule,
    }
    if definition.rule_type not in builders:
        raise ValueError(
            f"rule type '{definition.rule_type.value}' is not implemented yet "
            f"(available: {', '.join(t.value for t in builders)})"
        )
    return builders[definition.rule_type](definition)


class RuleSet:
    def __init__(self, definitions: list[RuleDefinition]) -> None:
        from packages.common.logging import get_logger

        log = get_logger(__name__)
        self.definitions = [d for d in definitions if d.enabled]
        self.rules: list[Rule] = []
        for definition in self.definitions:
            try:
                self.rules.append(build_rule(definition))
            except ValueError as exc:
                log.warning("rule skipped: %s", str(exc))

    def evaluate_all(self, ctx: EvaluationContext) -> list[RuleMatch]:
        from packages.common.logging import get_logger

        log = get_logger(__name__)
        matches: list[RuleMatch] = []
        for rule in self.rules:
            try:
                match = rule.evaluate(ctx)
            except Exception:
                log.exception("rule evaluation failed", extra={"rule": rule.definition.rule_id})
                continue
            if match is not None:
                matches.append(match)
        return matches
