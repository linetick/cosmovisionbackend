# Command Classification Guide

Этот файл описывает, как в backend добавлять новые команды управления 3D-моделью.

## Где находится логика

Основная логика классификации команд находится в файле:

- `app/query_logic.py`

Использование классификатора в API находится в файле:

- `app/api.py`

## Как сейчас работает классификация

Пайплайн такой:

1. Запрос нормализуется функцией `normalize_query`.
2. LLM-router в `classify_query_with_llm()` пытается определить:
   - `client_command`
   - `knowledge_answer`
   - `unknown_command`
3. Если LLM-router не дал валидный результат, backend использует fallback:
   - `detect_client_command()`
   - `looks_like_client_command()`
4. Если определён `client_command`, сервер возвращает JSON-команду клиенту.
5. Если это `knowledge_answer`, запрос идёт в RAG.

## Что нужно обновить при добавлении новой команды

### 1. Добавить фразы в `COMMAND_PHRASES`

В `app/query_logic.py` есть словарь:

```python
COMMAND_PHRASES = {
    ...
}
```

Нужно добавить новый ключ команды и список характерных фраз.

Пример:

```python
"stop_animation": (
    "останови анимацию",
    "выключи анимацию",
    "прекрати анимацию",
),
```

### 2. Добавить текст ответа в `COMMAND_ANSWERS`

В `app/query_logic.py` есть словарь:

```python
COMMAND_ANSWERS = {
    ...
}
```

Для новой команды нужно добавить человекочитаемый ответ.

Пример:

```python
"stop_animation": "Останавливаю анимацию спутника.",
```

### 3. Добавить маркеры в `COMMAND_LIKE_MARKERS`

В `app/query_logic.py` есть кортеж:

```python
COMMAND_LIKE_MARKERS = (
    ...
)
```

Сюда стоит добавить корни слов или устойчивые фразы, по которым запрос можно распознать как похожий на команду.

Пример:

```python
"анимац", "выключи анимацию", "прекрати анимацию"
```

Это важно для fallback-логики, если LLM-router не отработает корректно.

### 4. Обновить описание команд в `classify_query_with_llm()`

В `app/query_logic.py` функция `classify_query_with_llm()` содержит системный prompt.

Там нужно:

- добавить новую команду в текст описания;
- объяснить смысл команды;
- указать, какие формулировки к ней относятся.

Пример:

```text
- stop_animation: остановить анимацию, выключить движение, прекратить анимацию спутника.
```

### 5. Добавить few-shot примеры в `classify_query_with_llm()`

В той же функции есть список `messages` с примерами.

Нужно добавить хотя бы 1-2 новых примера для новой команды.

Пример:

```python
{
    "role": "user",
    "content": "Останови анимацию спутника",
},
{
    "role": "assistant",
    "content": '{"intent":"client_command","command_type":"stop_animation"}',
},
```

### 6. Убедиться, что клиент умеет обработать новую команду

Backend может вернуть новый `command_type`, но Unity-клиент тоже должен его понимать.

Например, если добавили:

```json
{
  "intent": "client_command",
  "client_command": {
    "type": "stop_animation"
  }
}
```

то клиентская часть должна содержать обработчик для `stop_animation`.

## Что обычно не нужно менять

Обычно не нужно трогать:

- `command_response()` в `app/api.py`
- `unknown_command_response()` в `app/api.py`

Они уже работают универсально для любых `command_type`, если команда есть в словарях.

## Как тестировать новую команду

После изменений:

1. Перезапустить сервер.
2. Проверить синтаксис:

```bash
python -m py_compile app/query_logic.py app/api.py server.py serverwithgrock.py
```

3. Проверить классификацию через debug endpoint:

```bash
curl -X POST http://127.0.0.1:8000/debug/command \
  -H "Content-Type: application/json" \
  -d '{"text":"Останови анимацию спутника"}'
```

Или через ngrok:

```bash
curl -X POST <BASE_URL>/debug/command \
  -H "Content-Type: application/json" \
  -d '{"text":"Останови анимацию спутника"}'
```

### Что смотреть в ответе

Полезные поля:

- `intent`
- `client_command`
- `command_debug.llm_route`
- `command_debug.rule_match`
- `command_debug.resolution`

Если всё настроено правильно, ответ должен быть таким:

```json
{
  "query": "Останови анимацию спутника",
  "intent": "client_command",
  "client_command": {
    "type": "stop_animation"
  },
  "answer": "Останавливаю анимацию спутника."
}
```

## Минимальный чек-лист добавления новой команды

При добавлении команды нужно обновить:

1. `COMMAND_PHRASES`
2. `COMMAND_ANSWERS`
3. `COMMAND_LIKE_MARKERS`
4. описание команды в `classify_query_with_llm()`
5. examples в `classify_query_with_llm()`
6. обработку новой команды на клиенте
