from datetime import date
from math import ceil


PHASE_RATIOS = {
    "base": 0.35,
    "build": 0.35,
    "peak": 0.20,
    "taper": 0.10,
}


def _phase_lengths(weeks_total: int) -> dict[str, int]:
    weeks_total = max(1, weeks_total)
    if weeks_total <= 4:
        return {"base": 1, "build": max(1, weeks_total - 2), "peak": 1, "taper": 1}

    base = max(1, round(weeks_total * PHASE_RATIOS["base"]))
    build = max(1, round(weeks_total * PHASE_RATIOS["build"]))
    peak = max(1, round(weeks_total * PHASE_RATIOS["peak"]))
    taper = max(1, weeks_total - base - build - peak)

    while base + build + peak + taper > weeks_total:
        if base >= build and base > 1:
            base -= 1
        elif build > 1:
            build -= 1
        elif peak > 1:
            peak -= 1
        else:
            taper -= 1

    while base + build + peak + taper < weeks_total:
        build += 1

    return {"base": base, "build": build, "peak": peak, "taper": taper}


def weeks_until(race_date: date, today: date | None = None) -> int:
    today = today or date.today()
    return max(1, ceil((race_date - today).days / 7))


def generate_training_plan(
    race_date: date,
    race_event: str,
    target_time_seconds: float,
    today: date | None = None,
) -> dict:
    weeks_total = weeks_until(race_date, today)
    phase_config = _phase_lengths(weeks_total)

    weekly_plans = []
    week_number = 1
    for phase, count in phase_config.items():
        for phase_week in range(1, count + 1):
            intensity = {
                "base": 0.68,
                "build": 0.78,
                "peak": 0.88,
                "taper": 0.58,
            }[phase]
            progression = 1 + ((phase_week - 1) * 0.05)
            target_load = round(28 * intensity * progression, 1)
            weekly_plans.append(
                {
                    "week_number": week_number,
                    "phase": phase,
                    "target_load": target_load,
                    "focus": _phase_focus(phase, race_event),
                    "sessions": _sessions_for_phase(phase, target_time_seconds),
                }
            )
            week_number += 1

    return {
        "weeks_total": weeks_total,
        "current_phase": weekly_plans[0]["phase"],
        "current_week": 1,
        "phase_config": phase_config,
        "weekly_plans": weekly_plans,
        "adaptation_log": [
            {
                "event": "created",
                "message": "Initial plan generated from race date, event, and target time.",
            }
        ],
    }


def _phase_focus(phase: str, race_event: str) -> str:
    focus = {
        "base": "Aerobic capacity and technical consistency",
        "build": "Threshold work and event-specific pace control",
        "peak": "Race-pace execution and high-quality speed",
        "taper": "Freshness, rhythm, starts, turns, and confidence",
    }
    return f"{focus[phase]} for {race_event}"


def _sessions_for_phase(phase: str, target_time_seconds: float) -> list[dict]:
    target_50 = round(target_time_seconds / 2, 2)
    templates = {
        "base": [
            {"type": "base", "main_set": "8x200 aerobic pull", "target": "Smooth technique under light fatigue"},
            {"type": "technique", "main_set": "12x50 drill/swim", "target": "Stable catch and body line"},
            {"type": "recovery", "main_set": "1600 easy mixed", "target": "Low RPE and clean turns"},
        ],
        "build": [
            {"type": "threshold", "main_set": "10x100 threshold", "target": "Hold repeatable pace"},
            {"type": "race_pace", "main_set": "16x25 at race rhythm", "target": f"Feel {target_50}s split rhythm"},
            {"type": "technique", "main_set": "8x75 build with video focus", "target": "Maintain stroke length"},
        ],
        "peak": [
            {"type": "race_pace", "main_set": "6x50 from blocks", "target": f"Hit {target_50}s split intent"},
            {"type": "vo2", "main_set": "12x25 fast on full rest", "target": "High speed without form loss"},
            {"type": "technique", "main_set": "Starts and turns quality set", "target": "Remove dead time"},
        ],
        "taper": [
            {"type": "race_pace", "main_set": "4x25 race feel", "target": "Fast, relaxed, precise"},
            {"type": "recovery", "main_set": "1200 easy with skills", "target": "Freshness and confidence"},
        ],
    }
    return templates[phase]
