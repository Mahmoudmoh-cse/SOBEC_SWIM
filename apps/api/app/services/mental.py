def generate_routine(values: dict[str, int]) -> dict:
    focus = values["mood_focus"]
    confidence = values["mood_confidence"]
    calm = values["mood_calm"]
    recovery = values["mood_recovery"]

    routine = {
        "breathing": "4 rounds of 4-second inhale, 6-second exhale",
        "visualization": "One perfect start, one clean turn, one composed finish",
        "cue_word": "long",
    }

    if confidence <= 5:
        routine["cue_word"] = "attack"
        routine["self_talk"] = "Name one training proof point before warm-up."
    elif calm <= 5:
        routine["cue_word"] = "settle"
        routine["self_talk"] = "Slow the first 15 seconds of the pre-race routine."
    elif recovery <= 5:
        routine["cue_word"] = "easy speed"
        routine["self_talk"] = "Warm up gradually and avoid forcing early speed."
    elif focus >= 8:
        routine["cue_word"] = "sharp"
        routine["self_talk"] = "Keep the routine short and protect the current focus state."
    else:
        routine["self_talk"] = "Choose one race detail to execute better than last time."

    return routine
