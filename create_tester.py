"""
Создаёт тестового суперпользователя с вечной подпиской и бессрочным токеном.
Запуск: python create_tester.py
"""
import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from jose import jwt
from app.auth.utils import hash_password, SECRET_KEY, ALGORITHM
from app.database import SessionLocal
from app.models import User, Subscription

TESTER_EMAIL    = "tester@cosmovision.test"
TESTER_PASSWORD = "cosmovision2026"

# Токен без срока истечения (поле exp отсутствует)
def create_eternal_token(user_id: int) -> str:
    return jwt.encode(
        {"sub": str(user_id), "type": "access"},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )

def main():
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == TESTER_EMAIL).first()
        if existing:
            token = create_eternal_token(existing.id)
            print(f"Пользователь уже существует (id={existing.id})")
            print(f"\nAccess token (бессрочный):\n{token}")
            return

        user = User(
            email=TESTER_EMAIL,
            password_hash=hash_password(TESTER_PASSWORD),
            tokens_used=0,
        )
        db.add(user)
        db.flush()

        # Подписка до 2099 года
        sub = Subscription(
            user_id=user.id,
            expires_at=datetime(2099, 12, 31, tzinfo=timezone.utc),
            is_active=True,
        )
        db.add(sub)
        db.commit()
        db.refresh(user)

        token = create_eternal_token(user.id)

        print(f"Тестовый пользователь создан!")
        print(f"  Email:    {TESTER_EMAIL}")
        print(f"  Password: {TESTER_PASSWORD}")
        print(f"  ID:       {user.id}")
        print(f"  Подписка: до 2099-12-31")
        print(f"\nAccess token (бессрочный):\n{token}")
        print(f"\nИспользование в Postman/запросах:")
        print(f"  Authorization: Bearer {token}")

    finally:
        db.close()

if __name__ == "__main__":
    main()
