# Model Storage

3D-модели хранятся в директории `model_storage/files/`.

Метаданные можно задавать в файле `model_storage/registry.json`.

Пример записи:

```json
[
  {
    "id": "meteor-m",
    "name": "Метеор-М",
    "description": "Метеорологический спутник.",
    "spacecraft": "Метеор-М",
    "format": "fbx",
    "file_name": "meteor_m.fbx",
    "relative_path": "meteor_m.fbx"
  }
]
```

Если `registry.json` пустой, сервер попробует автоматически найти модели по файлам
в `model_storage/files/` и сформировать минимальные метаданные по имени файла.
