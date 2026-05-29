from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.api.v1.utils import get_owned_swimmer
from app.db.session import get_db
from app.models import Swimmer, User
from app.schemas.common import SwimmerCreate, SwimmerRead, SwimmerUpdate

router = APIRouter(prefix="/swimmers", tags=["swimmers"])


@router.get("", response_model=list[SwimmerRead])
def list_swimmers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Swimmer]:
    return list(db.scalars(select(Swimmer).where(Swimmer.coach_id == current_user.id).order_by(Swimmer.name)))


@router.post("", response_model=SwimmerRead, status_code=201)
def create_swimmer(
    payload: SwimmerCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Swimmer:
    swimmer = Swimmer(coach_id=current_user.id, **payload.model_dump())
    db.add(swimmer)
    db.commit()
    db.refresh(swimmer)
    return swimmer


@router.get("/{swimmer_id}", response_model=SwimmerRead)
def get_swimmer(
    swimmer_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Swimmer:
    return get_owned_swimmer(db, swimmer_id, current_user)


@router.patch("/{swimmer_id}", response_model=SwimmerRead)
def update_swimmer(
    swimmer_id: str,
    payload: SwimmerUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Swimmer:
    swimmer = get_owned_swimmer(db, swimmer_id, current_user)
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(swimmer, key, value)
    db.commit()
    db.refresh(swimmer)
    return swimmer


@router.delete("/{swimmer_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_swimmer(
    swimmer_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    swimmer = get_owned_swimmer(db, swimmer_id, current_user)
    db.delete(swimmer)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
