# Industry rule packs

Rule packs configure SENTINEL behavior per industry — **configuration, not code**.
The engine in `services/rules` is identical for every industry.

Format: `packages/schemas/rule.py::RulePack` (validated on load by
`services/rules/pack.py`).

Implemented rule types: `zone_enter`, `zone_dwell`, `line_cross`.
Reserved for later phases: `object_count`, `proximity`, `temporal_signal`
(analyzers arrive with F6 - fall/violence/crowd candidates).

Zone IDs referenced by rules (`restricted`, `hazardous`, `loading`, `emergency`, …)
are operator-created zones bound per camera in the Zone Editor (phase F3).
