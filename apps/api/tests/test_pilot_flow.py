from datetime import date

from fastapi.testclient import TestClient


def _create_swimmer(client: TestClient, auth_headers: dict[str, str]) -> str:
    response = client.post(
        "/api/v1/swimmers",
        headers=auth_headers,
        json={
            "name": "Pilot Swimmer",
            "date_of_birth": "2010-02-02",
            "primary_stroke": "freestyle",
            "primary_event": "100m freestyle",
            "level": "age_group",
            "personal_bests": {"100m freestyle": 63.2},
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_pilot_board_returns_missing_data_flags(client: TestClient, auth_headers: dict[str, str]) -> None:
    swimmer_id = _create_swimmer(client, auth_headers)

    response = client.get("/api/v1/pilot/board", headers=auth_headers)

    assert response.status_code == 200
    board = response.json()
    card = next(item for item in board["swimmers"] if item["swimmer"]["id"] == swimmer_id)
    assert card["status"] == "needs_data"
    assert "log_session" in card["missing_flags"]
    assert board["totals"]["needs_session"] == 1


def test_pilot_session_entry_creates_session_and_checkin(client: TestClient, auth_headers: dict[str, str]) -> None:
    swimmer_id = _create_swimmer(client, auth_headers)

    response = client.post(
        "/api/v1/pilot/session-entry",
        headers=auth_headers,
        json={
            "swimmer_id": swimmer_id,
            "session_date": date.today().isoformat(),
            "session_type": "technique",
            "distance_m": 3000,
            "duration_min": 70,
            "rpe": 7,
            "mood_focus": 8,
            "mood_confidence": 7,
            "mood_energy": 7,
            "mood_calm": 7,
            "mood_recovery": 6,
            "mood_motivation": 8,
            "sleep_hours": 7.25,
            "notes": "First pilot entry.",
            "create_mental_checkin": True,
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["session"]["load_score"] == 21
    assert payload["mental_checkin"]["composite_score"] > 0
    assert payload["duplicate_warning"] is None
    assert "Saved 3000m technique session" in payload["entry_summary"]


def test_pilot_session_entry_warns_on_duplicate_without_blocking(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    swimmer_id = _create_swimmer(client, auth_headers)
    payload = {
        "swimmer_id": swimmer_id,
        "session_date": date.today().isoformat(),
        "session_type": "base",
        "distance_m": 2800,
        "duration_min": 65,
        "rpe": 6,
        "mood_focus": 7,
        "mood_confidence": 7,
        "mood_energy": 7,
        "mood_calm": 7,
        "mood_recovery": 7,
        "mood_motivation": 7,
        "create_mental_checkin": False,
    }

    first_response = client.post("/api/v1/pilot/session-entry", headers=auth_headers, json=payload)
    second_response = client.post("/api/v1/pilot/session-entry", headers=auth_headers, json=payload)

    assert first_response.status_code == 201
    assert second_response.status_code == 201
    assert second_response.json()["duplicate_warning"] is not None
    assert second_response.json()["session"]["session_type"] == "base"
