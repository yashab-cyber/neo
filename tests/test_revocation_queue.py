from fastapi.testclient import TestClient
from neo import create_app
import time


def test_async_command_execution():
    app = create_app()
    with TestClient(app) as client:
        def token(role="admin"):
            r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": role})
            assert r.status_code == 200
            return r.json()["access_token"], r.json()["refresh_token"]

        access, _ = token()
        r = client.post("/api/v1/commands/execute", json={"command": "do", "async_": True}, headers={"Authorization": f"Bearer {access}"})
        assert r.status_code == 200
        exec_id = r.json()["execution_id"]
        for _ in range(80):
            st = client.get(f"/api/v1/commands/{exec_id}", headers={"Authorization": f"Bearer {access}"})
            if st.json().get("status") in {"completed", "failed"}:
                assert st.json()["status"] == "completed"
                break
            time.sleep(0.025)
        else:
            assert False, "Command did not complete in time"


def test_token_revocation():
    app = create_app()
    with TestClient(app) as client:
        r = client.post("/api/v1/auth/token", json={"api_key": "devkey", "role": "admin"})
        assert r.status_code == 200
        access = r.json()["access_token"]
        refresh = r.json()["refresh_token"]
        r = client.post("/api/v1/auth/revoke", json={"refresh_token": refresh}, headers={"Authorization": f"Bearer {access}"})
        assert r.status_code == 200
        s = client.get("/api/v1/status", headers={"Authorization": f"Bearer {access}"})
        assert s.status_code == 401
