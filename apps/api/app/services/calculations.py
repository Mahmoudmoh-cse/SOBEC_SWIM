from statistics import mean


MOOD_FIELDS = (
    "mood_focus",
    "mood_confidence",
    "mood_energy",
    "mood_calm",
    "mood_recovery",
    "mood_motivation",
)


def calculate_session_load(distance_m: int, rpe: int) -> float:
    return round((distance_m * rpe) / 1000, 2)


def calculate_mental_composite(values: dict[str, int]) -> float:
    scores = [values[field] for field in MOOD_FIELDS]
    return round(mean(scores), 2)


def clamp_score(value: int | float, minimum: int = 0, maximum: int = 100) -> int:
    return int(max(minimum, min(maximum, round(value))))
