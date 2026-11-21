# Ответы Teacher A, B и C

Проверенная копия результатов генерации, завершённой 18 ноября 2026 года. Здесь сохранены исходные ответы, метрики и готовые входы общего `train.py`. При публикации ответы не генерировались заново и не редактировались.

| Teacher | Пилот: 100 задач | Полный набор: 1 000 задач |
| --- | --- | --- |
| A · Qwen2.5-1.5B-Instruct | [SFT JSONL](pilot/sft/teacher_A.jsonl) | [SFT JSONL](full/sft/teacher_A.jsonl) |
| B · Qwen2.5-3B-Instruct | [SFT JSONL](pilot/sft/teacher_B.jsonl) | [SFT JSONL](full/sft/teacher_B.jsonl) |
| C · Qwen2.5-7B-Instruct | [SFT JSONL](pilot/sft/teacher_C.jsonl) | [SFT JSONL](full/sft/teacher_C.jsonl) |

В каждой строке SFT есть `prompt_id`, `prompt` и `response`. Все три Teacher решали одинаковые задачи в одинаковом порядке. Пилот совпадает с первыми 100 строками полного набора. Исключённых пустых ответов нет; неверные ответы и ошибки формата сохранены.

Для C42 и C123 используется один и тот же файл Teacher C. Seed участник задаёт при обучении Student.

В папках [pilot](pilot) и [full](full) также находятся:

- `teacher_A.jsonl`, `teacher_B.jsonl`, `teacher_C.jsonl` — полные записи с условием, ответом, эталоном, параметрами модели, оценкой и временем генерации;
- `teacher_*.metrics.json` — метрики каждого Teacher;
- `sft/alignment.json` — число строк и общий список ID.

Все три исходных JSONL также доступны одним архивом: [пилот](pilot/raw.zip) и [полный набор](full/raw.zip). Содержимое архивов совпадает с отдельными JSONL.

| Teacher | Верных ответов на полном наборе |
| --- | --- |
| A | [358 из 1 000 (35,8%)](full/teacher_A.metrics.json) |
| B | [624 из 1 000 (62,4%)](full/teacher_B.metrics.json) |
| C | [568 из 1 000 (56,8%)](full/teacher_C.metrics.json) |

Если парсер не извлёк финальное число, задача учитывается как неверная.

Исходный [project_data.zip](../project_data.zip) содержит `train.jsonl` (1 000 задач), `validation.jsonl` (150) и `test.jsonl` (200). Контрольные суммы разбиений зафиксированы в [configs/teachers.json](../../configs/teachers.json). Условия взяты из GSM8K; [лицензия MIT](GSM8K-LICENSE.txt) сохранена вместе с данными.

## Обучение

Из корня репозитория после установки зависимостей:

```bash
python train.py --train-file data/teachers/full/sft/teacher_A.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --seed 123 --output-dir runs/A123/full
```

Для другого участника укажите нужный Teacher, согласованный seed и отдельную папку результатов. Это пример запуска общего тренера; полный протокол A123 с валидацией и оценкой описан в [основной инструкции](../../README.md). [Пакет B42](../../B42/README.md) и [сохранённые результаты B42](../../results/B42/README.md) доступны отдельно.

Для запуска A123 с готовыми ответами, валидацией и итоговой оценкой после установки зависимостей Student выполните из корня репозитория:

```bash
python -m zipfile -e data/teachers/full/raw.zip runs/full
python run_a123.py --stage full --data data/project_data.zip
```

Архив разбиений уже включён в репозиторий. Обёртка сверит ответы с задачами, подготовит вход SFT и запустит обучение и оценку A123.

## Проверка копии

[manifest.json](manifest.json) содержит размеры, SHA-256 и число строк всех опубликованных файлов данных. Команда из корня репозитория проверяет их без установки библиотек и загрузки моделей:

```bash
python -c "import hashlib,json,pathlib; m=json.loads(pathlib.Path('data/teachers/manifest.json').read_text(encoding='utf-8')); bad=[x['path'] for x in m['files'] if not pathlib.Path(x['path']).is_file() or pathlib.Path(x['path']).stat().st_size != x['bytes'] or hashlib.sha256(pathlib.Path(x['path']).read_bytes()).hexdigest() != x['sha256']]; print('Mismatches:', bad); raise SystemExit(bool(bad))"
```

Перед публикацией проверены все 3 000 отдельных записей кэша, оба набора SFT, метрики и исходные разбиения. Содержимое `runs/rows/` полностью представлено в полных JSONL. Локальные окружения, загруженные веса Teacher и временные рабочие файлы в этот снимок не входят.
