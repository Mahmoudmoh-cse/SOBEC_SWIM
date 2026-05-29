from datetime import date

from fastapi.testclient import TestClient


def test_delete_session_removes_wrong_entry(client: TestClient, auth_headers: dict[str, str]) -> None:
    swimmer = client.post(
        "/api/v1/swimmers",
        headers=auth_headers,
        json={
            "name": "Delete Session Swimmer",
            "date_of_birth": "2010-01-01",
            "primary_stroke": "freestyle",
            "primary_event": "100m freestyle",
            "level": "age_group",
            "personal_bests": {"100m freestyle": 62.1},
        },
    ).json()
    session = client.post(
        "/api/v1/sessions",
        headers=auth_headers,
        json={
            "swimmer_id": swimmer["id"],
            "session_date": date.today().isoformat(),
            "session_type": "base",
            "distance_m": 2400,
            "duration_min": 60,
            "rpe": 6,
            "mood_focus": 7,
            "mood_confidence": 7,
            "mood_energy": 7,
            "mood_calm": 7,
            "mood_recovery": 7,
            "mood_motivation": 7,
        },
    ).json()

    delete_response = client.delete(f"/api/v1/sessions/{session['id']}", headers=auth_headers)
    sessions_response = client.get(f"/api/v1/sessions?swimmer_id={swimmer['id']}", headers=auth_headers)

    assert delete_response.status_code == 204
    assert sessions_response.json() == []


def test_delete_swimmer_removes_profile_and_cascades_sessions(client: TestClient, auth_headers: dict[str, str]) -> None:
    swimmer = client.post(
        "/api/v1/swimmers",
        headers=auth_headers,
        json={
            "name": "Delete Swimmer",
            "date_of_birth": "2011-01-01",
            "primary_stroke": "backstroke",
            "primary_event": "100m backstroke",
            "level": "age_group",
            "personal_bests": {"100m backstroke": 68.4},
        },
    ).json()
    client.post(
        "/api/v1/sessions",
        headers=auth_headers,
        json={
            "swimmer_id": swimmer["id"],
            "session_date": date.today().isoformat(),
            "session_type": "technique",
            "distance_m": 2600,
            "duration_min": 65,
            "rpe": 6,
            "mood_focus": 7,
            "mood_confidence": 7,
            "mood_energy": 7,
            "mood_calm": 7,
            "mood_recovery": 7,
            "mood_motivation": 7,
        },
    )

    delete_response = client.delete(f"/api/v1/swimmers/{swimmer['id']}", headers=auth_headers)
    get_response = client.get(f"/api/v1/swimmers/{swimmer['id']}", headers=auth_headers)

    assert delete_response.status_code == 204
    assert get_response.status_code == 404
