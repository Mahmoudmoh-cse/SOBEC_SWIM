from statistics import mean

from app.services.calculations import clamp_score


def classify_strategy(splits: list[float]) -> str:
    if len(splits) < 2:
        return "custom"
    midpoint = len(splits) // 2
    first_half = sum(splits[:midpoint])
    second_half = sum(splits[midpoint:])
    delta = second_half - first_half
    if abs(delta) <= 0.25:
        return "even"
    return "negative" if delta < 0 else "positive"


def predicted_splits(total_time: float, count: int) -> list[float]:
    if count <= 0:
        return []
    split = round(total_time / count, 2)
    splits = [split for _ in range(count)]
    correction = round(total_time - sum(splits), 2)
    splits[-1] = round(splits[-1] + correction, 2)
    return splits


def strategy_score(actual: list[float], predicted: list[float]) -> int:
    if not actual or len(actual) != len(predicted):
        return 70
    avg_error = mean(abs(a - p) for a, p in zip(actual, predicted))
    return clamp_score(100 - (avg_error * 12))


def phase_analysis(actual: list[float], predicted: list[float]) -> dict:
    total_actual = round(sum(actual), 2)
    total_predicted = round(sum(predicted), 2)
    return {
        "actual_total": total_actual,
        "predicted_total": total_predicted,
        "delta_seconds": round(total_actual - total_predicted, 2),
        "fastest_split": min(actual) if actual else None,
        "slowest_split": max(actual) if actual else None,
    }


def generate_race_insights(strategy: str, score: int, time_vs_pb: float | None) -> list[str]:
    insights = []
    if strategy == "positive":
        insights.append("The race faded in the second half; rehearse controlled opening speed and stronger closing mechanics.")
    elif strategy == "negative":
        insights.append("The second half was stronger than the first; consider a slightly more assertive first split next race.")
    elif strategy == "even":
        insights.append("Pacing was stable; the next opportunity is reducing start, turn, or finish leakage.")
    else:
        insights.append("Split pattern is custom; compare against event plan before changing pacing.")

    if score < 80:
        insights.append("Execution drifted from the target model; add race-pace repeats with exact split feedback.")
    else:
        insights.append("Execution matched the model well; keep the pacing structure and chase technical gains.")

    if time_vs_pb is not None:
        if time_vs_pb < 0:
            insights.append(f"Personal-best improvement detected: {abs(time_vs_pb):.2f}s faster.")
        elif time_vs_pb > 0:
            insights.append(f"Time was {time_vs_pb:.2f}s off PB; isolate the largest split delta first.")
    return insights
