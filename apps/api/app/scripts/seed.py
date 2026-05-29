from datetime import date, timedelta

from sqlalchemy import select

from app.core.security import get_password_hash
from app.db.session import SessionLocal
from app.models import Session, Swimmer, TrainingPlan, User
from app.services.calculations import calculate_session_load
from app.services.training import generate_training_plan

DEMO_EMAIL = "coach@aquaiq.local"
DEMO_PASSWORD = "AquaIQ123!"


def seed() -> None:
    db = SessionLocal()
    try:
        coach = db.scalar(select(User).where(User.email == DEMO_EMAIL))
        if coach is None:
            coach = User(
                email=DEMO_EMAIL,
                full_name="AquaIQ Demo Coach",
                hashed_password=get_password_hash(DEMO_PASSWORD),
                role="coach",
            )
            db.add(coach)
            db.flush()

        swimmers = [
            Swimmer(
                coach_id=coach.id,
                name="Maya Hassan",
                date_of_birth=date(2009, 4, 12),
                primary_stroke="freestyle",
                primary_event="100m freestyle",
                level="age_group",
                personal_bests={"100m freestyle": 61.84},
                technique_profile={"catch": 76, "rotation": 81, "turn": 72},
                mental_profile={"confidence": 7.4, "focus": 8.1},
            ),
            Swimmer(
                coach_id=coach.id,
                name="Omar Khaled",
                date_of_birth=date(2007, 10, 3),
                primary_stroke="butterfly",
                primary_event="100m butterfly",
                level="junior",
                personal_bests={"100m butterfly": 58.42},
                technique_profile={"rhythm": 74, "kick": 70, "finish": 78},
                mental_profile={"confidence": 6.8, "calm": 6.2},
            ),
            Swimmer(
                coach_id=coach.id,
                name="Nour Saleh",
                date_of_birth=date(2008, 1, 27),
                primary_stroke="backstroke",
                primary_event="200m backstroke",
                level="elite",
                personal_bests={"200m backstroke": 133.2},
                technique_profile={"rotation": 83, "tempo": 79, "turn": 77},
                mental_profile={"motivation": 8.9, "recovery": 7.2},
            ),
        ]

        existing_names = set(db.scalars(select(Swimmer.name).where(Swimmer.coach_id == coach.id)))
        created_swimmers = []
        for swimmer in swimmers:
            if swimmer.name not in existing_names:
                db.add(swimmer)
                created_swimmers.append(swimmer)

        db.flush()
        target_swimmers = created_swimmers or list(db.scalars(select(Swimmer).where(Swimmer.coach_id == coach.id)))
        for index, swimmer in enumerate(target_swimmers):
            has_sessions = db.scalar(select(Session.id).where(Session.swimmer_id == swimmer.id).limit(1))
            if has_sessions:
                continue
            for offset, session_type in enumerate(["base", "technique", "race_pace"]):
                distance = 3200 + (index * 300) + (offset * 200)
                rpe = 5 + offset
                db.add(
                    Session(
                        swimmer_id=swimmer.id,
                        session_date=date.today() - timedelta(days=(offset * 2) + index),
                        session_type=session_type,
                        distance_m=distance,
                        duration_min=75 + offset * 8,
                        rpe=rpe,
                        load_score=calculate_session_load(distance, rpe),
                        mood_focus=7 + (offset % 2),
                        mood_confidence=7,
                        mood_energy=6 + offset,
                        mood_calm=7,
                        mood_recovery=6,
                        mood_motivation=8,
                        sleep_hours=7.5,
                        notes=f"Seeded {session_type} session for dashboard demo.",
                    )
                )

            active_plan = db.scalar(
                select(TrainingPlan.id).where(TrainingPlan.swimmer_id == swimmer.id, TrainingPlan.is_active.is_(True))
            )
            if active_plan is None:
                plan_data = generate_training_plan(
                    race_date=date.today() + timedelta(days=56),
                    race_event=swimmer.primary_event,
                    target_time_seconds=(next(iter(swimmer.personal_bests.values())) - 0.6),
                )
                db.add(
                    TrainingPlan(
                        swimmer_id=swimmer.id,
                        race_date=date.today() + timedelta(days=56),
                        race_event=swimmer.primary_event,
                        target_time_seconds=(next(iter(swimmer.personal_bests.values())) - 0.6),
                        **plan_data,
                    )
                )

        db.commit()
        print(f"Seeded AquaIQ demo data. Login: {DEMO_EMAIL} / {DEMO_PASSWORD}")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
