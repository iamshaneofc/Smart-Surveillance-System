"""F2-L authz verification: viewer/operator/admin across every F2 endpoint.

Read endpoints honor the read permissions, manage endpoints are admin-only,
event acknowledgements require operator+, evidence download requires
`evidence:export`, and denied attempts write no audit rows while admin
mutations are audited with the caller as actor.
"""

from packages.db import base as db_base, models

ADMIN = {"X-API-Key": "admin-key"}
OPERATOR = {"X-API-Key": "operator-key"}
VIEWER = {"X-API-Key": "viewer-key"}

SQUARE = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5], [0.1, 0.5]]


def _audit_rows():
    from sqlalchemy import select

    with db_base.session_scope() as session:
        return list(session.execute(select(models.AuditLog)).scalars().all())


def _camera_payload(camera_id="cam_authz"):
    return {
        "camera_id": camera_id,
        "name": f"Camera {camera_id}",
        "source_type": "synthetic",
        "stream_url": "",
    }


READ_MATRIX = [
    # path, expected: viewer, operator, admin
    ("/api/v1/cameras", 200, 200, 200),
    ("/api/v1/events", 200, 200, 200),
    ("/api/v1/evidence", 200, 200, 200),
    ("/api/v1/zones", 403, 200, 200),
    ("/api/v1/rules", 403, 200, 200),
    ("/api/v1/alerts", 403, 200, 200),
    ("/api/v1/system/health", 200, 200, 200),
    ("/api/v1/auth/me", 200, 200, 200),
]


def test_read_permission_matrix(api_key_client):
    for path, viewer_exp, operator_exp, admin_exp in READ_MATRIX:
        viewer = api_key_client.get(path, headers=VIEWER)
        assert viewer.status_code == viewer_exp, f"viewer {path}: {viewer.status_code}"
        operator = api_key_client.get(path, headers=OPERATOR)
        assert operator.status_code == operator_exp, f"operator {path}: {operator.status_code}"
        admin = api_key_client.get(path, headers=ADMIN)
        assert admin.status_code == admin_exp, f"admin {path}: {admin.status_code}"


def test_manage_endpoints_are_admin_only(api_key_client):
    mutations = [
        ("POST", "/api/v1/cameras", _camera_payload()),
        ("POST", "/api/v1/cameras/cam_authz/zones", {"name": "z", "zone_type": "restricted", "polygon": SQUARE}),
        ("POST", "/api/v1/rules", {"rule_id": "r1", "name": "R", "rule_type": "zone_enter", "event_type": "zone_enter", "severity": "high", "camera_id": "cam_authz", "zone_ids": ["z"]}),
        ("PATCH", "/api/v1/cameras/cam_authz", {"name": "renamed"}),
        ("DELETE", "/api/v1/cameras/cam_authz", None),
        ("PATCH", "/api/v1/zones/zone-none", {"name": "renamed"}),
        ("DELETE", "/api/v1/zones/zone-none", None),
        ("PATCH", "/api/v1/rules/r-none", {"name": "renamed"}),
        ("DELETE", "/api/v1/rules/r-none", None),
    ]
    for method, path, payload in mutations:
        for role, headers in (("viewer", VIEWER), ("operator", OPERATOR)):
            response = api_key_client.request(method, path, headers=headers, json=payload)
            assert response.status_code == 403, f"{role} {method} {path}: {response.status_code}"

    # denied attempts must not create audit rows
    assert _audit_rows() == []


def test_admin_mutations_are_audited(api_key_client):
    created = api_key_client.post("/api/v1/cameras", headers=ADMIN, json=_camera_payload())
    assert created.status_code == 201

    zone = api_key_client.post(
        "/api/v1/cameras/cam_authz/zones",
        headers=ADMIN,
        json={"name": "restricted", "zone_type": "restricted", "polygon": SQUARE},
    )
    assert zone.status_code == 201

    rule = api_key_client.post(
        "/api/v1/rules",
        headers=ADMIN,
        json={
            "rule_id": "perimeter-authz",
            "name": "Perimeter",
            "rule_type": "restricted_zone_intrusion",
            "event_type": "restricted_zone_intrusion",
            "severity": "high",
            "camera_id": "cam_authz",
            "zone_ids": [zone.json()["id"]],
        },
    )
    assert rule.status_code == 201

    patched = api_key_client.patch(
        "/api/v1/cameras/cam_authz", headers=ADMIN, json={"name": "renamed"}
    )
    assert patched.status_code == 200

    deleted = api_key_client.delete("/api/v1/cameras/cam_authz", headers=ADMIN)
    assert deleted.status_code == 204

    rows = _audit_rows()
    actions = [row.action for row in rows]
    for expected in ("camera.create", "zone.create", "rule.create", "camera.update", "camera.delete"):
        assert expected in actions, f"missing audit action {expected}"
    assert {row.actor for row in rows} == {"admin1"}


def test_event_status_requires_operator_or_admin(api_key_client):
    ack = api_key_client.post(
        "/api/v1/events/ev-none/status", headers=VIEWER, json={"status": "acknowledged"}
    )
    assert ack.status_code == 403

    dismiss = api_key_client.post(
        "/api/v1/events/ev-none/status", headers=VIEWER, json={"status": "dismissed"}
    )
    assert dismiss.status_code == 403

    # operator holds events:ack — passes authz, then 404 on the missing event
    operator_ack = api_key_client.post(
        "/api/v1/events/ev-none/status", headers=OPERATOR, json={"status": "acknowledged"}
    )
    assert operator_ack.status_code == 404

    admin_ack = api_key_client.post(
        "/api/v1/events/ev-none/status", headers=ADMIN, json={"status": "acknowledged"}
    )
    assert admin_ack.status_code == 404

    assert _audit_rows() == []


def test_evidence_download_requires_export_permission(api_key_client):
    viewer = api_key_client.get("/api/v1/evidence/ev-none/download", headers=VIEWER)
    assert viewer.status_code == 403

    operator = api_key_client.get("/api/v1/evidence/ev-none/download", headers=OPERATOR)
    assert operator.status_code == 403

    # admin passes authz; resource does not exist -> plain 404
    admin = api_key_client.get("/api/v1/evidence/ev-none/download", headers=ADMIN)
    assert admin.status_code == 404


def test_models_stub_is_admin_only(api_key_client):
    viewer = api_key_client.get("/api/v1/models", headers=VIEWER)
    assert viewer.status_code == 403
    operator = api_key_client.get("/api/v1/models", headers=OPERATOR)
    assert operator.status_code == 403
    admin = api_key_client.get("/api/v1/models", headers=ADMIN)
    assert admin.status_code == 501
