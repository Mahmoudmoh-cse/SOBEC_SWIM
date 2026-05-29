from fastapi import APIRouter

from app.api.v1 import (
    ai_outputs,
    auth,
    dashboard,
    mental_checkins,
    pilot,
    race_analyses,
    sessions,
    swimmers,
    technique_reports,
    training_plans,
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(ai_outputs.router)
api_router.include_router(auth.router)
api_router.include_router(dashboard.router)
api_router.include_router(pilot.router)
api_router.include_router(swimmers.router)
api_router.include_router(sessions.router)
api_router.include_router(technique_reports.router)
api_router.include_router(training_plans.router)
api_router.include_router(race_analyses.router)
api_router.include_router(mental_checkins.router)
