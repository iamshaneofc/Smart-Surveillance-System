from packages.db import base as db_base, models

ADMIN = {"X-API-Key": "admin-key"}
OPERATOR = {"X-API-Key": "operator-key"}
VIEWER = {"X-API-Key": "viewer-key"}


def _rtsp_payload(**overrides):
    payload = {
        "camera_id": "cam_f2a",
        "name": "Front gate",
        "location": "North gate",
        "source_type": "rtsp",
        "stream_url": "rtsp://user:secret@10.0.0.10/stream1",
        "metadata": {"floor": "1"},
    }
    payload.update(overrides)
    return payload


def _audit_actions():
    from sqlalchemy import select

    with db_base.session_scope() as session:
        rows = session.execute(select(models.AuditLog.action)).scalars().all()
    return list(rows)


def test_camera_create_get_list(client):
    created = client.post("/api/v1/cameras", json=_rtsp_payload())
    assert created.status_code == 201
    body = created.json()
    assert body["camera_id"] == "cam_f2a"
    assert body["stream_url_set"] is True
    assert body["metadata"] == {"floor": "1"}
    assert "stream_url" not in body

    detail = client.get("/api/v1/cameras/cam_f2a")
    assert detail.status_code == 200
    assert detail.json()["name"] == "Front gate"

    listing = client.get("/api/v1/cameras")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1

    assert "camera.create" in _audit_actions()


def test_camera_create_rejects_unsupported_sources(client):
    onvif = client.post(
        "/api/v1/cameras", json=_rtsp_payload(camera_id="cam_onvif", source_type="onvif")
    )
    assert onvif.status_code == 422
    assert onvif.json()["error"]["code"] == "validation_error"

    bad_rtsp = client.post(
        "/api/v1/cameras",
        json=_rtsp_payload(camera_id="cam_bad", stream_url="http://cam/stream"),
    )
    assert bad_rtsp.status_code == 422

    missing_file = client.post(
        "/api/v1/cameras",
        json=_rtsp_payload(camera_id="cam_file", source_type="file", stream_url=""),
    )
    assert missing_file.status_code == 422


def test_camera_synthetic_source_without_url(client):
    created = client.post(
        "/api/v1/cameras",
        json=_rtsp_payload(
            camera_id="cam_syn", source_type="synthetic", stream_url=""
        ),
    )
    assert created.status_code == 201
    assert created.json()["stream_url_set"] is False


def test_camera_create_duplicate_and_invalid_site(client):
    first = client.post("/api/v1/cameras", json=_rtsp_payload())
    assert first.status_code == 201

    duplicate = client.post("/api/v1/cameras", json=_rtsp_payload())
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "conflict"

    bad_site = client.post(
        "/api/v1/cameras", json=_rtsp_payload(camera_id="cam_site", site_id="site_missing")
    )
    assert bad_site.status_code == 422
    assert bad_site.json()["error"]["code"] == "invalid_site"


def test_camera_patch_updates_and_validates_stream_url(client):
    client.post("/api/v1/cameras", json=_rtsp_payload())

    patched = client.patch(
        "/api/v1/cameras/cam_f2a",
        json={"name": "Front gate 2", "enabled": False, "metadata": {"floor": "2"}},
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["name"] == "Front gate 2"
    assert body["enabled"] is False
    assert body["metadata"] == {"floor": "2"}

    invalid = client.patch(
        "/api/v1/cameras/cam_f2a", json={"stream_url": "http://nope"}
    )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_stream_url"

    valid = client.patch(
        "/api/v1/cameras/cam_f2a", json={"stream_url": "rtsp://10.0.0.11/live"}
    )
    assert valid.status_code == 200

    assert "camera.update" in _audit_actions()


def test_camera_soft_delete_preserves_history(client):
    client.post("/api/v1/cameras", json=_rtsp_payload())

    deleted = client.delete("/api/v1/cameras/cam_f2a")
    assert deleted.status_code == 204

    detail = client.get("/api/v1/cameras/cam_f2a")
    assert detail.status_code == 404

    listing = client.get("/api/v1/cameras")
    assert listing.json()["total"] == 0

    with_deleted = client.get("/api/v1/cameras/cam_f2a", params={"include_deleted": True})
    assert with_deleted.status_code == 200
    body = with_deleted.json()
    assert body["deleted_at"] is not None
    assert body["enabled"] is False

    recreate = client.post("/api/v1/cameras", json=_rtsp_payload())
    assert recreate.status_code == 409

    actions = _audit_actions()
    assert "camera.delete" in actions
    from sqlalchemy import select

    with db_base.session_scope() as session:
        row = session.execute(
            select(models.AuditLog).where(models.AuditLog.action == "camera.delete")
        ).scalar_one()
    assert row.details["mode"] == "soft_delete"


def test_camera_authz(api_key_client):
    created = api_key_client.post("/api/v1/cameras", json=_rtsp_payload(), headers=VIEWER)
    assert created.status_code == 403
    assert created.json()["error"]["code"] == "forbidden"

    created = api_key_client.post(
        "/api/v1/cameras", json=_rtsp_payload(camera_id="cam_op"), headers=OPERATOR
    )
    assert created.status_code == 403

    created = api_key_client.post(
        "/api/v1/cameras", json=_rtsp_payload(camera_id="cam_adm"), headers=ADMIN
    )
    assert created.status_code == 201

    patched = api_key_client.patch(
        "/api/v1/cameras/cam_adm", json={"name": "renamed"}, headers=OPERATOR
    )
    assert patched.status_code == 403

    listed = api_key_client.get("/api/v1/cameras", headers=VIEWER)
    assert listed.status_code == 200

    deleted = api_key_client.delete("/api/v1/cameras/cam_adm", headers=OPERATOR)
    assert deleted.status_code == 403

    deleted = api_key_client.delete("/api/v1/cameras/cam_adm", headers=ADMIN)
    assert deleted.status_code == 204
