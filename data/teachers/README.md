# Ответы Teacher A, B и C

Каждая модель решила одни и те же 1 000 задач GSM8K. Пилот содержит первые 100 задач полного набора. Для запуска C42 или C123 нужен один и тот же файл Teacher C. Seed участник задаёт при обучении Student.

| Teacher | Модель | Пилот, 100 задач | Полный набор, 1 000 задач | Верно на полном наборе |
| --- | --- | --- | --- | --- |
| A | Qwen2.5-1.5B-Instruct | [JSONL](pilot/sft/teacher_A.jsonl) | [JSONL](full/sft/teacher_A.jsonl) | [358 из 1 000 (35,8%)](full/teacher_A.metrics.json) |
| B | Qwen2.5-3B-Instruct | [JSONL](pilot/sft/teacher_B.jsonl) | [JSONL](full/sft/teacher_B.jsonl) | [624 из 1 000 (62,4%)](full/teacher_B.metrics.json) |
| C | Qwen2.5-7B-Instruct | [JSONL](pilot/sft/teacher_C.jsonl) | [JSONL](full/sft/teacher_C.jsonl) | [568 из 1 000 (56,8%)](full/teacher_C.metrics.json) |

В JSONL для обучения каждая строка содержит `prompt_id`, `prompt` и `response`. Передайте нужный файл в `--train-file` общего [train.py](../../train.py). Порядок задач одинаков у всех трёх Teacher. Списки ID доступны для [пилота](pilot/sft/alignment.json) и [полного набора](full/sft/alignment.json).

## Исходные ответы и метрики

В каждом ZIP лежат три исходных JSONL со всеми полями генератора. Для каждой задачи сохранены ответ модели, эталон, извлечённое число, флаг правильности, параметры генерации и время. Если парсер не извлёк финальное число, задача учитывается как неверная в таблице выше.

| Этап | Исходные ответы всех Teacher | Метрики по моделям |
| --- | --- | --- |
| Пилот | [Скачать ZIP](pilot/raw.zip) | [A](pilot/teacher_A.metrics.json) · [B](pilot/teacher_B.metrics.json) · [C](pilot/teacher_C.metrics.json) |
| Полный | [Скачать ZIP](full/raw.zip) | [A](full/teacher_A.metrics.json) · [B](full/teacher_B.metrics.json) · [C](full/teacher_C.metrics.json) |

## Запуск A123 с готовыми ответами

В свежей копии проекта положите исходный архив разбиений команды в `data/project_data.zip`. После установки зависимостей Student выполните из корня проекта две команды. Первая распакует ответы, вторая сверит их с задачами, подготовит вход SFT и запустит обучение и оценку A123.

```bash
python -m zipfile -e data/teachers/full/raw.zip runs/full
python run_a123.py --stage full --data data/project_data.zip
```

[Конфигурация Teacher](../../configs/teachers.json) · [Конфигурация A123](../../configs/A123.json) · [Лицензия исходных задач GSM8K](GSM8K-LICENSE.txt)
