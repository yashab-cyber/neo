from fastapi.testclient import TestClient
from neo import create_app

app = create_app()
client = TestClient(app)


def issue(role=None):
    payload = {"api_key": "devkey"}
    if role:
        payload["role"] = role
    r = client.post("/api/v1/auth/token", json=payload)
    assert r.status_code == 200
    return r.json()["access_token"]


def test_user_cannot_create_task():
    token = issue(role="user")
    r = client.post("/api/v1/tasks", json={"name": "t1", "command": "echo"}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code in (403, 401)


def test_admin_can_create_task():
    token = issue(role="admin")
    r = client.post("/api/v1/tasks", json={"name": "t1", "command": "echo"}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


def test_admin_execute_command():
    token = issue(role="admin")
    r = client.post("/api/v1/commands/execute", json={"command": "ping"}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


def test_user_cannot_execute_command():
    token = issue(role="user")
    r = client.post("/api/v1/commands/execute", json={"command": "ping"}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code in (403, 401)
