from packages.db import base as db_base, models

ADMIN = {"X-API-Key": "admin-key"}
OPERATOR = {"X-API-Key": "operator-key"}
VIEWER = {"X-API-Key": "viewer-key"}

SQUARE = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5], [0.1, 0.5]]


def _create_camera(client, camera_id="cam_zone"):
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


def _zone_payload(**overrides):
    payload = {
        "name": "Restricted area",
        "zone_type": "restricted",
        "polygon": SQUARE,
        "anchor": "center",
        "metadata": {"note": "no entry"},
    }
    payload.update(overrides)
    return payload


def _audit_actions():
    from sqlalchemy import select

    with db_base.session_scope() as session:
        return list(
            session.execute(select(models.AuditLog.action)).scalars().all()
        )


def test_zone_create_get_list(client):
    camera = _create_camera(client)

    created = client.post(
        f"/api/v1/cameras/{camera['camera_id']}/zones", json=_zone_payload()
    )
    assert created.status_code == 201
    zone = created.json()
    assert zone["camera_id"] == "cam_zone"
    assert zone["anchor"] == "center"
    assert zone["zone_type"] == "restricted"
    assert zone["metadata"] == {"note": "no entry"}

    detail = client.get(f"/api/v1/zones/{zone['id']}")
    assert detail.status_code == 200
    assert detail.json()["name"] == "Restricted area"

    under_camera = client.get("/api/v1/cameras/cam_zone/zones")
    assert under_camera.status_code == 200
    assert under_camera.json()["total"] == 1

    global_list = client.get("/api/v1/zones", params={"camera_id": "cam_zone"})
    assert global_list.status_code == 200
    assert global_list.json()["total"] == 1

    assert "zone.create" in _audit_actions()


def test_zone_rejects_invalid_geometry(client):
    camera = _create_camera(client)
    base = f"/api/v1/cameras/{camera['camera_id']}/zones"

    bowtie = client.post(
        base, json=_zone_payload(name="bowtie", polygon=[[0, 0], [1, 1], [0, 1], [1, 0]])
    )
    assert bowtie.status_code == 422
    assert bowtie.json()["error"]["code"] == "invalid_polygon"

    too_few = client.post(
        base, json=_zone_payload(name="line", polygon=[[0, 0], [1, 1]])
    )
    assert too_few.status_code == 422
    assert too_few.json()["error"]["code"] == "validation_error"

    out_of_bounds = client.post(
        base,
        json=_zone_payload(name="outside", polygon=[[0, 0], [1.5, 0], [1, 1]]),
    )
    assert out_of_bounds.status_code == 422
    assert out_of_bounds.json()["error"]["code"] == "validation_error"

    bad_anchor = client.post(base, json=_zone_payload(name="anchor", anchor="left"))
    assert bad_anchor.status_code == 422


def test_zone_duplicate_name_conflict(client):
    camera = _create_camera(client)
    base = f"/api/v1/cameras/{camera['camera_id']}/zones"

    first = client.post(base, json=_zone_payload())
    assert first.status_code == 201

    duplicate = client.post(base, json=_zone_payload())
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "conflict"

    _create_camera(client, camera_id="cam_zone2")
    other = client.post(
        "/api/v1/cameras/cam_zone2/zones", json=_zone_payload()
    )
    assert other.status_code == 201


def test_zone_missing_camera_returns_404(client):
    response = client.post(
        "/api/v1/cameras/nope/zones", json=_zone_payload()
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_zone_patch_and_delete(client):
    camera = _create_camera(client)
    created = client.post(
        f"/api/v1/cameras/{camera['camera_id']}/zones", json=_zone_payload()
    )
    zone_id = created.json()["id"]

    patched = client.patch(
        f"/api/v1/zones/{zone_id}",
        json={"name": "Renamed zone", "enabled": False, "anchor": "top_center"},
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["name"] == "Renamed zone"
    assert body["enabled"] is False
    assert body["anchor"] == "top_center"

    bad_patch = client.patch(
        f"/api/v1/zones/{zone_id}",
        json={"polygon": [[0, 0], [1, 1], [0, 1], [1, 0]]},
    )
    assert bad_patch.status_code == 422

    deleted = client.delete(f"/api/v1/zones/{zone_id}")
    assert deleted.status_code == 204

    missing = client.get(f"/api/v1/zones/{zone_id}")
    assert missing.status_code == 404

    actions = _audit_actions()
    assert "zone.update" in actions
    assert "zone.delete" in actions


def test_zone_authz(api_key_client):
    created = api_key_client.post(
        "/api/v1/cameras",
        json={
            "camera_id": "cam_az",
            "name": "Authz camera",
            "source_type": "synthetic",
            "stream_url": "",
        },
        headers=ADMIN,
    )
    assert created.status_code == 201

    denied = api_key_client.post(
        "/api/v1/cameras/cam_az/zones", json=_zone_payload(), headers=VIEWER
    )
    assert denied.status_code == 403

    denied = api_key_client.post(
        "/api/v1/cameras/cam_az/zones", json=_zone_payload(), headers=OPERATOR
    )
    assert denied.status_code == 403

    allowed = api_key_client.post(
        "/api/v1/cameras/cam_az/zones", json=_zone_payload(), headers=ADMIN
    )
    assert allowed.status_code == 201
    zone_id = allowed.json()["id"]

    listed = api_key_client.get("/api/v1/zones", headers=OPERATOR)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    no_access = api_key_client.get("/api/v1/zones", headers=VIEWER)
    assert no_access.status_code == 403

    patched = api_key_client.patch(
        f"/api/v1/zones/{zone_id}", json={"name": "op rename"}, headers=OPERATOR
    )
    assert patched.status_code == 403

    deleted = api_key_client.delete(f"/api/v1/zones/{zone_id}", headers=ADMIN)
    assert deleted.status_code == 204
