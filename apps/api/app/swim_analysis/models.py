from enum import StrEnum

from app.models import SwimAnalysis


class SwimAnalysisStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


__all__ = ["SwimAnalysis", "SwimAnalysisStatus"]
