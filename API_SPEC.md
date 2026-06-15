# CosmoVision Backend — API Specification

**Base URL:** `http://<host>:8000`  
**Формат:** JSON (Content-Type: application/json)  
**Авторизация:** Bearer токен в заголовке `Authorization: Bearer <access_token>`

---

## Авторизация

### POST /auth/register

Регистрация нового пользователя.

**Тело запроса:**
```json
{
  "email": "user@example.com",
  "password": "yourpassword"
}
```

**Ответ 201:**
```json
{
  "access_token": "<jwt>",
  "refresh_token": "<jwt>",
  "token_type": "bearer"
}
```

---

### POST /auth/login

Вход и получение токенов.

**Тело запроса:**
```json
{
  "email": "user@example.com",
  "password": "yourpassword"
}
```

**Ответ 200:**
```json
{
  "access_token": "<jwt>",
  "refresh_token": "<jwt>",
  "token_type": "bearer"
}
```

---

### POST /auth/refresh

Обновление access_token по refresh_token.

**Тело запроса:**
```json
{
  "refresh_token": "<jwt>"
}
```

**Ответ 200:**
```json
{
  "access_token": "<jwt>",
  "refresh_token": "<jwt>",
  "token_type": "bearer"
}
```

---

### POST /auth/logout

Инвалидация refresh_token. Требует авторизации.

**Ответ:** 204 No Content

---

### GET /auth/me

Информация о текущем пользователе. Требует авторизации.

**Ответ 200:**
```json
{
  "id": 1,
  "email": "user@example.com"
}
```

---

## Запросы (текст)

### POST /query

Основной эндпоинт. Принимает текстовый запрос пользователя, классифицирует намерение и возвращает ответ + команду для AR-сцены. Требует авторизации.

**Тело запроса:**
```json
{
  "text": "подсвети антенну",
  "current_model_id": "sputnik-1",
  "current_spacecraft": null
}
```

| Поле | Тип | Обязательно | Описание |
|------|-----|-------------|----------|
| `text` | string | да | Текст запроса пользователя |
| `current_model_id` | string\|null | нет | ID текущей 3D-модели из `/models`. Используется для резолва сущностей и контекста. |
| `current_spacecraft` | string\|null | нет | Имя КА текстом (альтернатива `current_model_id`) |
| `stream` | bool | нет | `false` по умолчанию. Если `true` — ответ возвращается как SSE-поток (см. ниже). |
| `session_id` | int\|null | нет | ID активной сессии. LLM получает историю переписки и отвечает с учётом контекста. |

**Ответ 200:**
```json
{
  "query": "подсвети антенну",
  "intent": "action",
  "client_command": {
    "type": "highlight_entity",
    "target_nodes": ["Bone_Antenna 1", "Bone_Antenna 2", "Bone_Antenna 3", "Bone_Antenna 4"]
  },
  "answer": "Подсвечиваю элемент.",
  "context_used": false,
  "timing": { "normalize": 0.12, "total": 1.77 }
}
```

---

#### Поле `intent`

| Значение | Описание |
|----------|----------|
| `action` | Команда управления 3D-моделью |
| `info` | Информационный запрос, ответ из базы знаний |
| `hybrid` | Команда + информационный вопрос одновременно |
| `unknown_command` | Пользователь хотел управлять моделью, но команда не распознана |
| `off_topic` | Запрос не по теме космонавтики |

---

#### Поле `client_command`

`null` для `intent: info`, `off_topic`, `unknown_command`.

Для `action` и `hybrid`:

```json
{
  "type": "<command_type>",
  "target_nodes": ["Bone_..."]
}
```

`target_nodes` присутствует только для `highlight_entity` и только если передан `current_model_id`.

#### Типы команд (`type`)

| Команда | Описание |
|---------|----------|
| `start_rotation` | Начать вращение модели |
| `stop_rotation` | Остановить вращение |
| `increase_scale` | Увеличить масштаб |
| `decrease_scale` | Уменьшить масштаб |
| `reset_view` | Сбросить вид в исходное состояние |
| `play_animation` | Запустить анимацию |
| `highlight_entity` | Подсветить указанный элемент модели |

---

#### Примеры ответов по типу intent

**action — обычная команда:**
```json
{
  "query": "вращай спутник",
  "intent": "action",
  "client_command": { "type": "start_rotation" },
  "answer": "Запускаю вращение спутника.",
  "context_used": false
}
```

**action — highlight_entity:**
```json
{
  "query": "подсвети передатчик",
  "intent": "action",
  "client_command": {
    "type": "highlight_entity",
    "target_nodes": ["Bone_Transmitter"]
  },
  "answer": "Подсвечиваю элемент.",
  "context_used": false
}
```

**info:**
```json
{
  "query": "что такое спутник-1",
  "intent": "info",
  "client_command": null,
  "answer": "Спутник-1 — первый в мире искусственный спутник Земли...",
  "context_used": true
}
```

**hybrid — команда + вопрос:**
```json
{
  "query": "подсвети батарею и расскажи о ней",
  "intent": "hybrid",
  "client_command": {
    "type": "highlight_entity",
    "target_nodes": ["Bone_Battery"]
  },
  "knowledge_query": "расскажи о батарее",
  "answer": "Подсвечиваю элемент. Батарея обеспечивала питание бортовой аппаратуры...",
  "context_used": true
}
```

**unknown_command:**
```json
{
  "query": "сделай что-нибудь со спутником",
  "intent": "unknown_command",
  "client_command": null,
  "answer": "Команда не распознана.",
  "context_used": false
}
```

**off_topic:**
```json
{
  "query": "как дела",
  "intent": "off_topic",
  "client_command": null,
  "answer": "Я отвечаю только по космонавтике из базы знаний...",
  "context_used": false
}
```

---

#### Режим стриминга (`stream: true`)

Если передать `"stream": true`, ответ возвращается как `text/event-stream` (SSE) — текст приходит частями по мере генерации.

**Поток событий:**

```
data: {"type":"meta","intent":"info","client_command":null}

data: {"type":"token","text":"Спутник"}
data: {"type":"token","text":"-1 —"}
data: {"type":"token","text":" первый в мире..."}

data: {"type":"done"}
```

| `type` | Описание |
|--------|----------|
| `meta` | Первое событие. Содержит `intent` и `client_command`. Клиент сразу выполняет AR-команду, не дожидаясь конца текста. |
| `token` | Фрагмент текста. Добавлять к UI последовательно. |
| `done` | Стрим завершён. |

**Пример — action:**
```
data: {"type":"meta","intent":"action","client_command":{"type":"highlight_entity","target_nodes":["Bone_Antenna 1","Bone_Antenna 2"]}}

data: {"type":"token","text":"Подсвечиваю элемент."}

data: {"type":"done"}
```

**Пример — hybrid:**
```
data: {"type":"meta","intent":"hybrid","client_command":{"type":"highlight_entity","target_nodes":["Bone_Battery"]}}

data: {"type":"token","text":"Подсвечиваю элемент. "}
data: {"type":"token","text":"Батарея обеспечивала питание..."}

data: {"type":"done"}
```

---

### POST /query_audio

Голосовой запрос. Принимает аудиофайл, транскрибирует через Whisper и обрабатывает как текстовый запрос. Требует авторизации.

**Тело запроса:** `multipart/form-data`

| Поле | Тип | Описание |
|------|-----|----------|
| `file` | binary | Аудиофайл (wav, ogg, m4a, mp3 и др.) |
| `current_model_id` | string\|null | ID текущей 3D-модели |
| `current_spacecraft` | string\|null | Имя КА текстом |
| `stream` | bool | `false` по умолчанию. Если `true` — ответ возвращается как SSE-поток. |
| `session_id` | int\|null | ID активной сессии для передачи истории переписки в LLM. |

**Ответ 200 (stream: false)** — структура аналогична `/query`, дополнительно:
```json
{
  "transcript": "подсвети антенну",
  "intent": "action",
  "client_command": { ... },
  "answer": "...",
  "timing": {
    "asr": 0.85,
    "transcribe": 0.80,
    "normalize": 0.01,
    "total": 2.63
  }
}
```

**Ответ (stream: true)** — SSE-поток, аналогичный `/query`. Транскрипт передаётся в первом событии `meta`:

```
data: {"type":"meta","intent":"action","client_command":{"type":"highlight_entity","target_nodes":["Bone_Antenna 1"]},"transcript":"подсвети антенну"}

data: {"type":"token","text":"Подсвечиваю элемент."}

data: {"type":"done"}
```

---

## Сессии (история чата)

Сессия хранит историю переписки. При передаче `session_id` в `/query` или `/query_audio` LLM получает предыдущие сообщения и может отвечать на follow-up вопросы («а какой у него был диаметр?», «когда его запустили?»). Требуют авторизации.

### POST /sessions

Создать новую сессию.

**Query-параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `current_model_id` | string\|null | Привязать сессию к 3D-модели |

**Ответ 201:**
```json
{
  "id": 1,
  "spacecraft_id": "sputnik-1",
  "created_at": "2026-06-15T01:46:38.601933"
}
```

---

### GET /sessions

Список сессий текущего пользователя (последние 50, по убыванию даты).

**Ответ 200:**
```json
{
  "items": [
    {
      "id": 1,
      "spacecraft_id": "sputnik-1",
      "created_at": "2026-06-15T01:46:38",
      "updated_at": "2026-06-15T02:10:00",
      "last_message": "когда его запустили?"
    }
  ],
  "count": 1
}
```

---

### GET /sessions/{id}/messages

Полная история сообщений сессии.

**Ответ 200:**
```json
{
  "session_id": 1,
  "spacecraft_id": "sputnik-1",
  "messages": [
    { "id": 1, "role": "user",      "content": "сколько весил спутник-1?", "intent": "info", "created_at": "..." },
    { "id": 2, "role": "assistant", "content": "Масса Спутника-1 составляла 83,6 кг.", "intent": "info", "created_at": "..." },
    { "id": 3, "role": "user",      "content": "а какой у него был диаметр?", "intent": "info", "created_at": "..." },
    { "id": 4, "role": "assistant", "content": "Диаметр сферического корпуса — 58 см.", "intent": "info", "created_at": "..." }
  ]
}
```

**Ответ 404:** сессия не найдена или не принадлежит пользователю.

---

### DELETE /sessions/{id}

Удалить сессию вместе со всей историей.

**Ответ:** 204 No Content  
**Ответ 404:** сессия не найдена.

---

#### Типичный сценарий использования

```
1. POST /sessions                          → { "id": 5 }
2. POST /query { session_id: 5, text: "что такое спутник-1?" }
3. POST /query { session_id: 5, text: "а сколько он весил?" }   ← LLM помнит контекст
4. POST /query { session_id: 5, text: "когда его запустили?" }  ← и этот тоже
5. GET  /sessions/5/messages               → вся история
6. DELETE /sessions/5                      → очистить
```

---

## 3D-модели

### GET /models

Список всех доступных 3D-моделей.

**Ответ 200:**
```json
{
  "items": [
    {
      "id": "sputnik-1",
      "name": "Спутник-1",
      "description": "Первый искусственный спутник Земли...",
      "spacecraft": "Спутник-1",
      "format": "fbx",
      "file_name": "Sputnik-1.fbx",
      "relative_path": "Sputnik-1.fbx",
      "scene": {
        "root_node": "Sputnik-1001",
        "entities": {
          "antenna": {
            "nodes": ["Bone_Antenna 1", "Bone_Antenna 2", "Bone_Antenna 3", "Bone_Antenna 4"],
            "aliases": ["антенна", "антенны", "ус", "усы"]
          }
        }
      }
    }
  ],
  "count": 1,
  "storage_dir": "/path/to/model_storage"
}
```

---

### GET /models/{model_id}

Метаданные конкретной модели.

**Параметры пути:** `model_id` — ID модели (например, `sputnik-1`)

**Ответ 200:** один объект из массива `items` выше.  
**Ответ 404:** модель не найдена.

---

### GET /models/{model_id}/download

Скачать файл 3D-модели.

**Параметры пути:** `model_id` — ID модели

**Ответ 200:** бинарный файл (`application/octet-stream`)  
**Ответ 404:** файл не найден.

---

## Коды ошибок

| Код | Описание |
|-----|----------|
| 401 | Не авторизован / токен истёк |
| 404 | Ресурс не найден |
| 409 | Email уже занят (при регистрации) |
| 500 | Внутренняя ошибка сервера |

---

## Поле `scene` и `target_nodes`

Поле `scene` в метаданных модели описывает структуру 3D-сцены. Для команды `highlight_entity` сервер автоматически разрешает название детали из запроса в список узлов (`target_nodes`) — клиенту нужно подсветить именно эти узлы в сцене.

```
Пользователь: "подсвети антенну"
     ↓
Сервер находит alias "антенна" → entity "antenna" → nodes ["Bone_Antenna 1", ...]
     ↓
client_command: { "type": "highlight_entity", "target_nodes": ["Bone_Antenna 1", ...] }
     ↓
Клиент подсвечивает узлы по именам в FBX-сцене
```

Если `target_nodes` отсутствует в ответе — сущность не удалось определить или `current_model_id` не был передан.
