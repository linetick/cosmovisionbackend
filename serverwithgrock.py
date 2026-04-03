import os
import threading
import time

import nest_asyncio
import uvicorn
from pyngrok import conf, ngrok

from app.api import app


NGROK_TOKEN = os.getenv(
    "NGROK_TOKEN",
    "39jhsX0Tw6Kp5vbEwB8xqZeaHbs_81R3cLX6c7bpau5HLpypn",
)
conf.get_default().auth_token = NGROK_TOKEN


def run_server() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    nest_asyncio.apply()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    time.sleep(3)
    print("\n🌐 Проброс порта через ngrok...")

    try:
        public_url = ngrok.connect(8000)
        ngrok_url = str(public_url)

        print("\n" + "=" * 60)
        print("🚀 СЕРВЕР РАБОТАЕТ!")
        print("=" * 60)
        print(f"🔗 Swagger UI: {ngrok_url}/docs")
        print(f"🔗 Debug KB: {ngrok_url}/debug/kb")
        print(f"🔗 Models: {ngrok_url}/models")
        print("=" * 60)
        print("Ожидание подключений (не закрывайте эту ячейку)...\n")

        while True:
            time.sleep(1)
    except Exception as exc:
        print(f"Ошибка ngrok: {exc}")
        while True:
            time.sleep(1)
