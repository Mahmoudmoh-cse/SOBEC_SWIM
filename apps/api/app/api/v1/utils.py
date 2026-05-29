from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models import Swimmer, User


def get_owned_swimmer(db: Session, swimmer_id: str, user: User) -> Swimmer:
    swimmer = db.get(Swimmer, swimmer_id)
    if swimmer is None or swimmer.coach_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Swimmer not found")
    return swimmer
