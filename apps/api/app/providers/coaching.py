from abc import ABC, abstractmethod


class CoachingTextGenerator(ABC):
    @abstractmethod
    def technique_summary(self, swimmer_name: str, top_fault: str, score: int, time_gain: float) -> str:
        raise NotImplementedError


class LocalCoachingTextGenerator(CoachingTextGenerator):
    def technique_summary(self, swimmer_name: str, top_fault: str, score: int, time_gain: float) -> str:
        if score >= 84:
            tone = "Your stroke is trending well."
        elif score >= 72:
            tone = "You are close to a cleaner stroke."
        else:
            tone = "The next technical jump is clear."
        return (
            f"{tone} {swimmer_name}, the main limiter today is {top_fault}. "
            f"Clean that up in the next session and the model estimates about {time_gain:.2f}s available."
        )


def get_coaching_generator() -> CoachingTextGenerator:
    return LocalCoachingTextGenerator()
