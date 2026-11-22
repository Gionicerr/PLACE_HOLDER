# B42 — Student на ответах Teacher B

[Повторный запуск от 22 ноября 2026 года](rerun/) входит в [актуальную тройку A42, B42 и C42](../README.md). Для новой оценки загружай адаптер из `results/B42/rerun/adapter`. Адаптер первого запуска от 18 ноября находится в `results/B42/adapter`. Именно его использовали для оценки B42 в [PR № 2](https://github.com/Gionicerr/PLACE_HOLDER/pull/2).

Участник № 3 обучил `Qwen/Qwen2.5-0.5B-Instruct` с `seed=42` на 1 000 ответах Teacher B (`Qwen2.5-3B-Instruct`, GGUF Q4_K_M). Для обучения использовался общий [train.py](../../train.py), для входных данных и запуска — [пакет B42 из ветки участника № 2](https://github.com/Gionicerr/PLACE_HOLDER/tree/bae8237b5782fda37563dccb86254858a620e1a1/B42).

| Показатель | Результат |
| --- | ---: |
| Эпохи и шаги оптимизатора | 2 эпохи, 126 шагов |
| Train loss | 0,2305 |
| Validation loss на 150 задачах | 2,8204 |
| Время обучения на Tesla T4 | 538,6 с |
| Пиковая видеопамять, выделенная PyTorch | 5,31 ГиБ |
| Test accuracy | Ожидает оценки |

Точные значения сохранены в [run_results.json](run_results.json). [run_config.json](run_config.json) фиксирует ревизию модели, параметры и версии библиотек. [trainer_state.json](trainer_state.json) содержит историю шагов и loss.

## Журнал и заметки участника № 3

18 ноября 2026 года участник № 3 лично запустил B42 в Google Colab. В первой попытке пробный запуск остановился на импорте с ошибкой `torchvision::nms`: установленный в Colab `torchvision` был рассчитан на другую версию PyTorch. После установки `torchvision==0.22.1` для CUDA 12.6 пробный запуск выполнил один шаг на пилотных данных. Полное обучение стартовало отдельным запуском с исходного checkpoint и теми же настройками модели и LoRA.

В полном запуске использован опубликованный файл `teacher_B_full.jsonl` из пакета № 2. Его SHA-256 — `df52c93bfbed4a93d4eec55b6a18c60713b025fa9146e615d0076c4a1df3ab8d`. Запуск на одной Tesla T4 дошёл до 126-го шага и завершил две эпохи. LoRA-адаптер сохранён в [adapter/](adapter/). По этому журналу и конфигурации можно сверить параметры при повторе эксперимента.

Train loss измерялся на полных ответах Teacher B, а validation loss — на эталонных строках с финальным числом. Для вывода о качестве B42 нужны доля правильных ответов на общем тесте и сравнение с baseline. Эти метрики рассчитает участник № 4 по единому протоколу оценки.

## Передача участнику № 4

Используй 200 задач из `test.jsonl` командного архива `project_data.zip`. SHA-256 файла внутри архива — `d615995981e42a14e132cba14974637668c6344a356fee953584b658225fe138`. Запусти baseline и B42 одним скриптом на тех же задачах с одинаковыми промптами, настройками генерации и парсером ответов.

Для сопоставимости с A123 после условия задачи добавляй два перевода строки и общую инструкцию: `Solve the math problem. Show a short calculation. Finish with a separate line containing #### followed by the final number, with no units or additional text.` Используй шаблон диалога из токенизатора, жадное декодирование, лимит 192 новых токенов и числовую проверку финальной строки `#### число`. Правило разбора есть в [protocol.py участника № 2](https://github.com/Gionicerr/PLACE_HOLDER/blob/bae8237b5782fda37563dccb86254858a620e1a1/protocol.py).

Установи зависимости из [requirements.txt](../../requirements.txt) в отдельное окружение Python. Этот пример загружает адаптер из корня репозитория:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter = "results/B42/rerun/adapter"
tokenizer = AutoTokenizer.from_pretrained(adapter, padding_side="left")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    use_safetensors=True,
)
model = PeftModel.from_pretrained(base, adapter).eval()
```

Для baseline выполни генерацию в контексте `model.disable_adapter()`, для B42 — с активным адаптером. В итоговую таблицу внеси число верных ответов, test accuracy обеих моделей и разницу в процентных пунктах.
