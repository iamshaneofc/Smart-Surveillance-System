ADMIN = {"X-API-Key": "admin-key"}
OPERATOR = {"X-API-Key": "operator-key"}
VIEWER = {"X-API-Key": "viewer-key"}


def _create_camera(client, camera_id="cam_rule"):
    response = client.post(
        "/api/v1/cameras",
        json={
            "camera_id": camera_id,
            "name": f"Camera {camera_id}",
            "source_type": "synthetic",
            "stream_url": "",
        },
    )
    assert response.status_code == 201
    return response.json()


def _rule_payload(**overrides):
    payload = {
        "rule_id": "perimeter-enter",
        "name": "Perimeter enter",
        "rule_type": "zone_enter",
        "event_type": "zone_enter",
        "severity": "high",
        "zone_ids": ["z-perimeter"],
        "params": {"dwell_seconds": 0},
        "cooldown_seconds": 30,
        "camera_id": "cam_rule",
    }
    payload.update(overrides)
    return payload


def test_rule_create_get_list(client):
    _create_camera(client)

    created = client.post("/api/v1/rules", json=_rule_payload())
    assert created.status_code == 201
    body = created.json()
    assert body["rule_id"] == "perimeter-enter"
    assert body["version"] == "1"
    assert body["camera_id"] == "cam_rule"
    assert body["schedule"]["windows"] == []
    assert "created_at" in body and "updated_at" in body

    detail = client.get("/api/v1/rules/perimeter-enter")
    assert detail.status_code == 200
    assert detail.json()["severity"] == "high"

    listing = client.get("/api/v1/rules")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1

    filtered = client.get("/api/v1/rules", params={"rule_type": "line_cross"})
    assert filtered.json()["total"] == 0

    filtered = client.get("/api/v1/rules", params={"camera_id": "cam_rule"})
    assert filtered.json()["total"] == 1


def test_rule_rejects_unsupported_type(client):
    _create_camera(client)
    response = client.post(
        "/api/v1/rules", json=_rule_payload(rule_id="count-rule", rule_type="object_count")
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unsupported_rule_type"


def test_rule_conflicts_and_scope_validation(client):
    _create_camera(client)
    first = client.post("/api/v1/rules", json=_rule_payload())
    assert first.status_code == 201

    duplicate = client.post("/api/v1/rules", json=_rule_payload())
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "conflict"

    missing_camera = client.post(
        "/api/v1/rules", json=_rule_payload(rule_id="other-rule", camera_id="nope")
    )
    assert missing_camera.status_code == 404

    bad_site = client.post(
        "/api/v1/rules",
        json=_rule_payload(rule_id="site-rule", camera_id=None, site_id="site_missing"),
    )
    assert bad_site.status_code == 422
    assert bad_site.json()["error"]["code"] == "invalid_site"


def test_rule_patch_bumps_version_and_locks_id(client):
    _create_camera(client)
    created = client.post("/api/v1/rules", json=_rule_payload())
    assert created.json()["version"] == "1"

    patched = client.patch(
        "/api/v1/rules/perimeter-enter", json={"name": "Perimeter entry v2", "cooldown_seconds": 45}
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["name"] == "Perimeter entry v2"
    assert body["cooldown_seconds"] == 45
    assert body["version"] == "2"

    rename = client.patch("/api/v1/rules/perimeter-enter", json={"rule_id": "renamed"})
    assert rename.status_code == 409
    assert rename.json()["error"]["code"] == "rule_id_immutable"

    again = client.patch("/api/v1/rules/perimeter-enter", json={"enabled": False})
    assert again.json()["version"] == "3"

    no_change = client.patch("/api/v1/rules/perimeter-enter", json={"enabled": False})
    assert no_change.json()["version"] == "3"


def test_rule_delete(client):
    _create_camera(client)
    client.post("/api/v1/rules", json=_rule_payload())

    deleted = client.delete("/api/v1/rules/perimeter-enter")
    assert deleted.status_code == 204

    missing = client.get("/api/v1/rules/perimeter-enter")
    assert missing.status_code == 404

    listing = client.get("/api/v1/rules")
    assert listing.json()["total"] == 0


def test_rule_authz(api_key_client):
    created = api_key_client.post(
        "/api/v1/cameras",
        json={
            "camera_id": "cam_rauth",
            "name": "Rule camera",
            "source_type": "synthetic",
            "stream_url": "",
        },
        headers=ADMIN,
    )
    assert created.status_code == 201

    denied = api_key_client.post(
        "/api/v1/rules", json=_rule_payload(camera_id="cam_rauth"), headers=OPERATOR
    )
    assert denied.status_code == 403
    denied = api_key_client.post(
        "/api/v1/rules", json=_rule_payload(camera_id="cam_rauth"), headers=VIEWER
    )
    assert denied.status_code == 403

    allowed = api_key_client.post(
        "/api/v1/rules",
        json=_rule_payload(rule_id="authz-rule", camera_id="cam_rauth"),
        headers=ADMIN,
    )
    assert allowed.status_code == 201
    rule_id = allowed.json()["rule_id"]

    listed = api_key_client.get("/api/v1/rules", headers=OPERATOR)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    no_read = api_key_client.get("/api/v1/rules", headers=VIEWER)
    assert no_read.status_code == 403

    patched = api_key_client.patch(
        f"/api/v1/rules/{rule_id}", json={"enabled": False}, headers=OPERATOR
    )
    assert patched.status_code == 403

    deleted = api_key_client.delete(f"/api/v1/rules/{rule_id}", headers=ADMIN)
    assert deleted.status_code == 204
