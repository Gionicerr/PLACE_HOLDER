# B42 · обучение на ответах Teacher B

[**Скачать B42-project.zip**](https://raw.githubusercontent.com/Gionicerr/PLACE_HOLDER/task2/teacher-generation-a-seed123/B42/B42-project.zip) · [Открыть страницу файла на GitHub](https://github.com/Gionicerr/PLACE_HOLDER/blob/task2/teacher-generation-a-seed123/B42/B42-project.zip)

В архиве лежат оба JSONL, общий `train.py` и код проекта. Распакуйте ZIP и откройте терминал в папке с `train.py`. Со страницы файла архив можно скачать кнопкой **Download raw file**.

| Набор | Строк | Скачать отдельно |
| --- | ---: | --- |
| Пилот для проверки и пробного обучения | 100 | [teacher_B_pilot.jsonl](https://raw.githubusercontent.com/Gionicerr/PLACE_HOLDER/task2/teacher-generation-a-seed123/B42/teacher_B_pilot.jsonl) |
| Полный train для B42 | 1 000 | [teacher_B_full.jsonl](https://raw.githubusercontent.com/Gionicerr/PLACE_HOLDER/task2/teacher-generation-a-seed123/B42/teacher_B_full.jsonl) |

Пилот — первые 100 строк полного набора. Если уже скачан `teacher_B_pilot_check.jsonl`, это тот же файл. Каждая строка содержит `prompt_id`, `prompt` и `response`. Ответы сгенерировала Qwen2.5-3B-Instruct (GGUF Q4_K_M). ID и порядок задач проверены по всем трём Teacher.

## Проверка данных

Достаточно Python. Каждая команда проверяет число строк, порядок ID, непустые поля и SHA-256 файла.

```sh
python -m B42.run pilot --check
python -m B42.run full --check
```

## Обучение

В активном окружении Python 3.11 или 3.12 установите зависимости общего тренера. При первом запуске обучения понадобится интернет для загрузки Student. Ответы Teacher уже лежат в архиве.

```sh
python -m pip install -r requirements-a123.txt
python -m B42.run smoke
python -m B42.run full
```

`smoke` выполняет один шаг оптимизатора на пилотных данных. `pilot` обучает на всех 100 примерах две эпохи. `full` запускает независимое обучение на 1 000 примерах с исходного checkpoint. Для полного запуска можно сразу выполнить последнюю команду.

Все режимы вызывают командный [train.py](../train.py) с `seed=42`. Student — `Qwen/Qwen2.5-0.5B-Instruct`, ревизия `7ae557604adf67be50417f59c2c2f167def9a775`. Полный запуск использует две эпохи, LoRA rank 8, alpha 16, learning rate 0.0002 и длину до 512 токенов. На одном устройстве эффективный размер батча равен 16.

Если нужен validation loss, передайте архив участника № 1 при запуске. Для полного набора используются 150 валидационных задач, для пилота — 30.

```sh
python -m B42.run full --data data/project_data.zip
```

После полного запуска адаптер появится в `runs/B42/full/adapter/`, метрики — в `runs/B42/full/run_results.json`, параметры запуска — в `runs/B42/full/run_config.json`. Для повтора укажите новую папку через `--output-dir`. Для принудительного запуска на CPU добавьте `--cpu`.

Условия задач взяты из командной выборки GSM8K. [Исходные задачи](https://github.com/openai/grade-school-math) · [Лицензия MIT](GSM8K-LICENSE.txt)
