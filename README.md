# PLACE_HOLDER

## Общее обучение Student

Все участники запускают [train.py](train.py) с одним исходным checkpoint Student. Для каждого эксперимента укажи файл ответов своего Teacher, seed и отдельную папку результатов.

Установи зависимости в отдельное окружение Python 3.11 или 3.12. Первый запуск требует доступа к Hugging Face для загрузки базовой модели.

```bash
python -m pip install -r requirements.txt
```

Обучающий JSONL содержит непустые строки `prompt` и `response`. Дополнительные поля, например `prompt_id`, можно сохранить. Используй согласованные в команде наборы SFT с одинаковыми ID задач у всех Teacher.

```bash
python train.py \
  --train-file data/teacher_B_full.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --seed 42 \
  --output-dir runs/B42/full
```

По умолчанию используются две эпохи, LoRA rank 8 и alpha 16, learning rate `2e-4` и длина последовательности до 512 токенов. На одном устройстве эффективный размер батча равен 16. Скрипт применяет шаблон диалога модели и считает loss по ответу ассистента. В папке запуска появляются `adapter/`, `run_config.json`, `run_results.json` и `trainer_state.json`.

Для пробного запуска возьми пилотный JSONL, добавь `--max-steps 1` и выбери новую папку результатов. Для расчёта validation loss передай `--validation-file` с теми же полями. По протоколу A123/B42 промпт валидации содержит задачу и общую инструкцию, а `response` содержит `#### ` и эталонное число из validation. Тестовую выборку используй только на этапе итоговой оценки.

Файлы Teacher B и команда для точного повтора B42 с валидацией есть в [пакете B42](https://github.com/Gionicerr/PLACE_HOLDER/tree/bae8237b5782fda37563dccb86254858a620e1a1/B42). Адаптер, журнал и инструкция по передаче на оценку лежат в [results/B42](results/B42).
