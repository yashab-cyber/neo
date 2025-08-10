from fastapi.testclient import TestClient
from neo import create_app

client = TestClient(create_app())


def _token():
    r = client.post('/api/v1/auth/token', json={'api_key': 'devkey'})
    assert r.status_code == 200
    return r.json()['access_token']


def test_chat():
    token = _token()
    r = client.post('/api/v1/ai/chat', json={'message': 'Hello world'}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    data = r.json()
    assert 'response' in data
    assert data['response'].startswith('Stub response')
