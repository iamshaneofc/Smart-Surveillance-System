# Edge deployment — deferred

Target: single on-site GPU/CPU box running camera workers + inference + API,
local-disk evidence storage, fully functional without internet (no raw video to cloud).

Planned artifacts: hardened image, systemd/container autostart, resource limits,
offline update bundle. See `docs/DEVELOPMENT_ROADMAP.md` (F7).
