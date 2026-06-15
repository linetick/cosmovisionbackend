from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ..config import AUTH_REQUIRED
from ..database import get_db
from ..models import User
from .utils import decode_token

bearer_scheme = HTTPBearer(auto_error=AUTH_REQUIRED)

_ANONYMOUS = User(id=0, email="anonymous")


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if not AUTH_REQUIRED:
        if credentials is None:
            return _ANONYMOUS
        # Токен передан — всё равно проверим, но ошибку не бросаем
        try:
            payload = decode_token(credentials.credentials)
            user_id = payload.get("sub")
            if user_id and payload.get("type") == "access":
                user = db.get(User, int(user_id))
                if user:
                    return user
        except ValueError:
            pass
        return _ANONYMOUS

    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Токен не передан")

    try:
        payload = decode_token(credentials.credentials)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Недействительный токен")

    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Ожидается access token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Недействительный токен")

    user = db.get(User, int(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Пользователь не найден")

    return user
