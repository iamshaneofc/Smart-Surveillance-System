from packages.schemas.common import Severity
from packages.schemas.event import ConditionEvidence
from packages.schemas.rule import RuleDefinition
from services.rules.context import EvaluationContext
from services.rules.geometry import crossing_direction
from services.rules.schedule import is_active


def _cond(name, operator, actual, threshold, satisfied=True) -> ConditionEvidence:
    return ConditionEvidence(
        name=name, operator=operator, actual=actual, threshold=threshold, satisfied=satisfied
    )


def _schedule_condition(rule_def: RuleDefinition, ctx: EvaluationContext) -> ConditionEvidence | None:
    if not rule_def.schedule.windows:
        return None
    window = rule_def.schedule.windows[0]
    return _cond(
        "schedule",
        "in_window",
        ctx.ts.isoformat(),
        f"{window.start}-{window.end}",
        True,
    )


def _classes(rule_def: RuleDefinition) -> list[str]:
    return list(rule_def.params.get("classes", ["person"]))


class ZoneEnterRule:
    def __init__(self, definition: RuleDefinition) -> None:
        self.definition = definition

    def evaluate(self, ctx: EvaluationContext) -> "RuleMatch | None":
        d = self.definition
        if not is_active(d.schedule, ctx.ts, ctx.camera_timezone):
            return None
        zone_filter = set(d.zone_ids)
        classes = _classes(d)
        for tr in ctx.transitions:
            if tr.kind != "enter":
                continue
            if zone_filter and tr.zone_id not in zone_filter:
                continue
            if tr.class_name not in classes:
                continue
            if tr.confidence < d.min_confidence:
                continue
            conditions = [
                _cond("object_class", "in", tr.class_name, ",".join(classes)),
                _cond("zone", "in", tr.zone_name, ",".join(sorted(zone_filter)) or "any"),
                _cond("confidence", ">=", tr.confidence, d.min_confidence),
            ]
            schedule_cond = _schedule_condition(d, ctx)
            if schedule_cond is not None:
                conditions.append(schedule_cond)
            from services.rules.base import RuleMatch

            return RuleMatch(
                rule_key=d.rule_id,
                rule_name=d.name,
                event_type=d.event_type,
                severity=d.severity,
                camera_id=ctx.camera_id,
                ts=tr.ts,
                confidence=tr.confidence,
                track_ids=[tr.track_id],
                zone_id=tr.zone_id,
                zone_name=tr.zone_name,
                conditions=conditions,
                metadata={"transition": tr.kind},
            )
        return None


class ZoneDwellRule:
    def __init__(self, definition: RuleDefinition) -> None:
        self.definition = definition

    def evaluate(self, ctx: EvaluationContext) -> "RuleMatch | None":
        d = self.definition
        if not is_active(d.schedule, ctx.ts, ctx.camera_timezone):
            return None
        if ctx.spatial is None:
            return None
        dwell_seconds = float(d.params.get("dwell_seconds", 20.0))
        zone_filter = set(d.zone_ids)
        classes = _classes(d)
        for item in ctx.tracks:
            track = item.track
            if track.class_name not in classes or track.confidence < d.min_confidence:
                continue
            for zone_id in item.zone_ids:
                if zone_filter and zone_id not in zone_filter:
                    continue
                dwell = ctx.spatial.dwell_seconds(track.track_id, zone_id, ctx.ts)
                if dwell is None or dwell < dwell_seconds:
                    continue
                zone = next((z for z in ctx.zones if z.id == zone_id), None)
                zone_name = zone.name if zone else zone_id
                from services.rules.base import RuleMatch

                conditions = [
                    _cond("object_class", "in", track.class_name, ",".join(classes)),
                    _cond("zone", "in", zone_name, ",".join(sorted(zone_filter)) or "any"),
                    _cond("dwell_seconds", ">=", round(dwell, 1), dwell_seconds),
                    _cond("confidence", ">=", track.confidence, d.min_confidence),
                ]
                schedule_cond = _schedule_condition(d, ctx)
                if schedule_cond is not None:
                    conditions.append(schedule_cond)
                return RuleMatch(
                    rule_key=d.rule_id,
                    rule_name=d.name,
                    event_type=d.event_type,
                    severity=d.severity,
                    camera_id=ctx.camera_id,
                    ts=ctx.ts,
                    confidence=track.confidence,
                    track_ids=[track.track_id],
                    zone_id=zone_id,
                    zone_name=zone_name,
                    conditions=conditions,
                    metadata={"dwell_seconds": round(dwell, 1)},
                )
        return None


class RestrictedZoneIntrusionRule:
    """F1 core rule: a matching object stays inside a configured restricted polygon.

    Emits a match on every evaluation frame while the track is inside the zone.
    Temporal confirmation (confirmation/confirm_seconds) and cooldown are enforced
    by the EventEngine, so the effective semantics are:

        person in zone AND confidence >= min_confidence
        sustained for >= confirm_seconds   =>  EVENT
    """

    def __init__(self, definition: RuleDefinition) -> None:
        self.definition = definition

    def evaluate(self, ctx: EvaluationContext) -> "RuleMatch | None":
        d = self.definition
        if not is_active(d.schedule, ctx.ts, ctx.camera_timezone):
            return None
        if ctx.spatial is None:
            return None
        zone_filter = set(d.zone_ids)
        classes = _classes(d)
        best: "RuleMatch | None" = None
        for item in ctx.tracks:
            track = item.track
            if track.class_name not in classes or track.confidence < d.min_confidence:
                continue
            for zone_id in item.zone_ids:
                if zone_filter and zone_id not in zone_filter:
                    continue
                dwell = ctx.spatial.dwell_seconds(track.track_id, zone_id, ctx.ts)
                if dwell is None:
                    continue
                zone = next((z for z in ctx.zones if z.id == zone_id), None)
                zone_name = zone.name if zone else zone_id
                entered = ctx.spatial.zone_since(track.track_id, zone_id)
                from services.rules.base import RuleMatch

                conditions = [
                    _cond("object_class", "in", track.class_name, ",".join(classes)),
                    _cond("zone", "in", zone_name, ",".join(sorted(zone_filter)) or "any"),
                    _cond("confidence", ">=", track.confidence, d.min_confidence),
                    _cond(
                        "in_zone_dwell_seconds",
                        ">=",
                        round(dwell, 1),
                        f"confirmed after {d.confirm_seconds}s",
                    ),
                ]
                schedule_cond = _schedule_condition(d, ctx)
                if schedule_cond is not None:
                    conditions.append(schedule_cond)
                candidate = RuleMatch(
                    rule_key=d.rule_id,
                    rule_name=d.name,
                    event_type=d.event_type,
                    severity=d.severity,
                    camera_id=ctx.camera_id,
                    ts=ctx.ts,
                    confidence=track.confidence,
                    track_ids=[track.track_id],
                    zone_id=zone_id,
                    zone_name=zone_name,
                    conditions=conditions,
                    metadata={
                        "dwell_seconds": round(dwell, 1),
                        "entered_zone_at": entered.isoformat() if entered else None,
                        "track_state": track.state,
                    },
                )
                if best is None or candidate.confidence > best.confidence:
                    best = candidate
        return best


class LineCrossRule:
    def __init__(self, definition: RuleDefinition) -> None:
        self.definition = definition

    def evaluate(self, ctx: EvaluationContext) -> "RuleMatch | None":
        d = self.definition
        if not is_active(d.schedule, ctx.ts, ctx.camera_timezone):
            return None
        if ctx.spatial is None or not d.line or len(d.line) != 2:
            return None
        wanted_direction = d.params.get("direction")
        classes = _classes(d)
        for item in ctx.tracks:
            track = item.track
            if track.class_name not in classes or track.confidence < d.min_confidence:
                continue
            prev = ctx.spatial.previous_center(track.track_id)
            if prev is None:
                continue
            direction = crossing_direction(prev, item.center, d.line[0], d.line[1])
            if direction is None:
                continue
            if wanted_direction and direction != wanted_direction:
                continue
            from services.rules.base import RuleMatch

            conditions = [
                _cond("object_class", "in", track.class_name, ",".join(classes)),
                _cond("line_crossing", "==", direction, wanted_direction or "any"),
                _cond("confidence", ">=", track.confidence, d.min_confidence),
            ]
            schedule_cond = _schedule_condition(d, ctx)
            if schedule_cond is not None:
                conditions.append(schedule_cond)
            return RuleMatch(
                rule_key=d.rule_id,
                rule_name=d.name,
                event_type=d.event_type,
                severity=d.severity,
                camera_id=ctx.camera_id,
                ts=ctx.ts,
                confidence=track.confidence,
                track_ids=[track.track_id],
                zone_id=None,
                zone_name=None,
                conditions=conditions,
                metadata={"direction": direction},
            )
        return None
